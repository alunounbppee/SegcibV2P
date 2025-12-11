#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coleta cenários de pedestre atravessando FAIXA de pedestre com carro se aproximando.

- Específico para CARLA 0.9.15, mapa Town03.
- Pressupõe servidor já rodando em modo headless (-nullrhi -nosound).
- Veículo em autopilot + TrafficManager, pedestre cruzando a faixa.

Estrutura gerada (compatível com anota_radar_pedestre.py):

data/carla/dataset_clean/<run_id>/
  radar/
    radar_XXXXXX.csv    (azimuth, altitude, depth, velocity)
  events/
    crosswalk.csv       (frame, ts, veh_id, walker_id, walker_type, distance, cw_idx)
"""

import argparse
import math
import random
import time
import queue
import inspect
from datetime import datetime
from pathlib import Path

import carla

CROSSWALK_RADIUS = 4.0     # raio quando só temos Location de crosswalk
MAX_EVENT_DIST = 40.0      # distância máx. carro–pedestre para registrar evento
RADAR_PPS = 4000           # pontos por segundo do radar


# --------------------------- helpers geom / IO -----------------------

def dist(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def in_bbox2d(loc: carla.Location,
              bbox: carla.BoundingBox,
              tr: carla.Transform,
              margin: float = 0.75) -> bool:
    dx = loc.x - tr.location.x
    dy = loc.y - tr.location.y
    yaw = math.radians(tr.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    ex, ey = bbox.extent.x + margin, bbox.extent.y + margin
    return (-ex <= lx <= ex) and (-ey <= ly <= ey)


def get_crosswalks(world: carla.World):
    crosswalk_bbs, crosswalk_locs = [], []
    try:
        cw_raw = world.get_map().get_crosswalks() or []
        for obj in cw_raw:
            if hasattr(obj, "extent"):
                crosswalk_bbs.append(obj)
            else:
                crosswalk_locs.append(obj)
    except Exception:
        pass
    return crosswalk_bbs, crosswalk_locs


def nearest_spawn_to(world: carla.World, target: carla.Location):
    sps = world.get_map().get_spawn_points()
    if not sps:
        return None
    return min(sps, key=lambda sp: sp.location.distance(target))


def random_nav_location(world: carla.World):
    for _ in range(100):
        loc = world.get_random_location_from_navigation()
        if loc:
            return loc
    return None


def _lane_mask_sidewalk_crosswalk():
    mask = carla.LaneType.Sidewalk
    try:
        mask = mask | getattr(carla.LaneType, "Crosswalk")
    except Exception:
        pass
    return mask


def find_sidewalk_wp(world: carla.World, near: carla.Location):
    m = world.get_map()
    lane_mask = _lane_mask_sidewalk_crosswalk()
    wp = m.get_waypoint(near, project_to_road=True, lane_type=lane_mask)
    if wp:
        return wp

    for r in (2.0, 4.0, 6.0, 8.0):
        for _ in range(10):
            ang = random.uniform(0, 2 * math.pi)
            probe = carla.Location(
                x=near.x + r * math.cos(ang),
                y=near.y + r * math.sin(ang),
                z=near.z,
            )
            wp = m.get_waypoint(probe, project_to_road=True, lane_type=lane_mask)
            if wp:
                return wp
    return None


def resilient_spawn_walker(world: carla.World,
                           walker_bp: carla.ActorBlueprint,
                           prefer_loc: carla.Location):
    # 1) tenta perto da crosswalk
    if prefer_loc:
        wp = find_sidewalk_wp(world, prefer_loc)
        if wp:
            base = wp.transform.location
            for dz in (0.8, 1.0, 1.2):
                tr = carla.Transform(carla.Location(base.x, base.y, base.z + dz))
                w = world.try_spawn_actor(walker_bp, tr)
                if w:
                    return w

    # 2) local aleatório de navegação
    for _ in range(80):
        loc = random_nav_location(world)
        if not loc:
            continue
        for dz in (0.8, 1.0, 1.2):
            tr = carla.Transform(carla.Location(loc.x, loc.y, loc.z + dz))
            w = world.try_spawn_actor(walker_bp, tr)
            if w:
                return w

    # 3) fallback em spawn point
    sps = world.get_map().get_spawn_points()
    if sps:
        sp = sps[0].location
        for dz in (0.8, 1.0, 1.2):
            tr = carla.Transform(carla.Location(sp.x, sp.y, sp.z + dz))
            w = world.try_spawn_actor(walker_bp, tr)
            if w:
                return w
    return None


# ----------------------- principal: coleta episódios ------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    default_root = project_root / "data" / "carla" / "dataset_clean"

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=400,
                    help="Frames por episódio (default: 400)")
    ap.add_argument("--episodes", type=int, default=20,
                    help="Número de episódios (veículo+pedestre) por execução")
    ap.add_argument("--run-id", default=None,
                    help="Nome da subpasta do dataset. Se vazio, usa timestamp AAAAMMDD_HHMMSS.")
    args = ap.parse_args()

    print(">>> carla module:", inspect.getfile(carla))
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    print(">>> Server:", client.get_server_version())

    # Sempre cria subpasta nova (NUNCA grava direto em dataset_clean/)
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = default_root / run_id
    rad_dir = out / "radar"
    evt_dir = out / "events"
    rad_dir.mkdir(parents=True, exist_ok=True)
    evt_dir.mkdir(parents=True, exist_ok=True)

    # se já existe crosswalk.csv nessa subpasta, aborta pra não sobrescrever
    cross_path = evt_dir / "crosswalk.csv"
    if cross_path.exists():
        raise RuntimeError(f"{cross_path} já existe, escolha outro --run-id.")

    cross_f = cross_path.open("w", buffering=1)
    cross_f.write("frame,ts,veh_id,walker_id,walker_type,distance,cw_idx\n")

    try:
        world = client.load_world("Town10")
    except Exception:
        world = client.get_world()

    s = world.get_settings()
    orig_sync, orig_norend, orig_fps = (
        s.synchronous_mode,
        s.no_rendering_mode,
        s.fixed_delta_seconds,
    )
    s.synchronous_mode = True
    s.no_rendering_mode = True
    s.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(s)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(True)
    tm.global_percentage_speed_difference(10)

    bp = world.get_blueprint_library()
    veh_bp = (bp.filter("vehicle.*model3*") or bp.filter("vehicle.*"))[0]
    walker_bp = random.choice(bp.filter("walker.pedestrian.*"))

    radar_bp = bp.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", "40")
    radar_bp.set_attribute("vertical_fov", "30")
    radar_bp.set_attribute("range", "60")
    radar_bp.set_attribute("points_per_second", str(RADAR_PPS))
    radar_bp.set_attribute("sensor_tick", str(1.0 / args.fps))

    crosswalk_bbs, crosswalk_locs = get_crosswalks(world)
    sps = world.get_map().get_spawn_points()

    print(f">>> crosswalk_bbs: {len(crosswalk_bbs)}, crosswalk_locs: {len(crosswalk_locs)}")
    if not crosswalk_bbs and not crosswalk_locs:
        print("Não encontrei crosswalks no mapa, saindo.")
        return

    radar_queue: queue.Queue = queue.Queue()

    try:
        for _ in range(10):
            world.tick()

        for ep in range(args.episodes):
            print(f"\n=== Episódio {ep+1}/{args.episodes} ===")

            # escolhe faixa alvo
            if crosswalk_bbs:
                cw_bb = random.choice(crosswalk_bbs)
                cw_loc = cw_bb.location
            else:
                cw_bb = None
                cw_loc = random.choice(crosswalk_locs)

            # veículo perto da faixa
            chosen_sp = nearest_spawn_to(world, cw_loc) or random.choice(sps)
            vehicle = world.try_spawn_actor(veh_bp, chosen_sp) \
                or world.spawn_actor(veh_bp, random.choice(sps))
            vehicle.set_autopilot(True, tm.get_port())

            # pedestre perto da faixa
            walker = resilient_spawn_walker(world, walker_bp, prefer_loc=cw_loc)
            if not walker:
                raise RuntimeError("Falha ao spawnar pedestre (após tentativas)")

            # direção de travessia (aprox. perpendicular à faixa)
            if cw_bb and hasattr(cw_bb, "rotation"):
                yaw_rad = math.radians(cw_bb.rotation.yaw + 90.0)
                walk_dir = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
            else:
                walk_dir = carla.Vector3D(1.0, 0.0, 0.0)

            walk_speed = random.uniform(1.0, 2.0)
            walker.apply_control(carla.WalkerControl(direction=walk_dir, speed=walk_speed))

            # radar no carro
            radar = world.spawn_actor(
                radar_bp,
                carla.Transform(carla.Location(x=0.8, z=1.6)),
                attach_to=vehicle,
            )

            # limpa fila antiga
            while not radar_queue.empty():
                try:
                    radar_queue.get_nowait()
                except queue.Empty:
                    break

            def on_radar(meas: carla.RadarMeasurement):
                try:
                    radar_queue.put_nowait(meas)
                except queue.Full:
                    pass

            radar.listen(on_radar)

            for _ in range(5):
                world.tick()

            for _ in range(args.frames):
                world.tick()
                snapshot = world.get_snapshot()
                fr = snapshot.frame
                ts = time.time()

                try:
                    meas = radar_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                rad_path = rad_dir / f"radar_{fr:06d}.csv"
                with rad_path.open("w") as rf:
                    rf.write("azimuth,altitude,depth,velocity\n")
                    for d in meas:
                        rf.write(
                            f"{d.azimuth:.6f},{d.altitude:.6f},"
                            f"{d.depth:.3f},{d.velocity:.3f}\n"
                        )

                wloc = walker.get_location()
                vloc = vehicle.get_location()
                dveh = dist(wloc, vloc)

                # checar se pedestre está na faixa
                cw_idx = -1
                if crosswalk_bbs:
                    for i, bb in enumerate(crosswalk_bbs):
                        yaw = getattr(getattr(bb, "rotation", None), "yaw", 0.0)
                        tr = carla.Transform(bb.location, carla.Rotation(yaw=yaw))
                        if in_bbox2d(wloc, bb, tr, margin=0.75):
                            cw_idx = i
                            break
                else:
                    for i, loc in enumerate(crosswalk_locs):
                        if dist(wloc, loc) <= CROSSWALK_RADIUS:
                            cw_idx = i
                            break

                if dveh <= MAX_EVENT_DIST and cw_idx >= 0:
                    cross_f.write(
                        f"{fr},{ts},{vehicle.id},{walker.id},"
                        f"{walker.type_id},{dveh:.3f},{cw_idx}\n"
                    )

            # cleanup deste episódio
            try:
                radar.listen(lambda *_a, **_k: None)
                radar.stop()
            except Exception:
                pass
            for a in (radar, walker, vehicle):
                try:
                    if a:
                        a.destroy()
                except Exception:
                    pass

            print(f"Episódio {ep+1} finalizado.")

        print(f"\nOK: dados em {out}")

    finally:
        try:
            cross_f.close()
        except Exception:
            pass
        try:
            s = world.get_settings()
            s.synchronous_mode = orig_sync
            s.no_rendering_mode = orig_norend
            s.fixed_delta_seconds = orig_fps
            world.apply_settings(s)
            tm.set_synchronous_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
