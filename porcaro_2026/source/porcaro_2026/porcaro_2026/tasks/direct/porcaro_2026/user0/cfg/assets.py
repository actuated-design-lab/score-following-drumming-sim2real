from __future__ import annotations
from pathlib import Path
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import MassPropertiesCfg


# 初期関節角（sim の座標系。下向きが正）
WRIST_J0       = math.radians(0.0)
GRIP_J0        = math.radians(-8.1)


def quat_from_euler_zyx(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    # 入力は度。Z（ヨー）→ Y（ピッチ）→ X（ロール）の順。返り値は (w, x, y, z)
    z = math.radians(yaw_deg)
    y = math.radians(pitch_deg)
    x = math.radians(roll_deg)
    cz, sz = math.cos(z/2), math.sin(z/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    cx, sx = math.cos(x/2), math.sin(x/2)
    # q = qz * qy * qx（右手系）
    w =  cz*cy*cx + sz*sy*sx
    qx = cz*cy*sx - sz*sy*cx
    qy = cz*sy*cx + sz*cy*sx
    qz = sz*cy*cx - cz*sy*sx
    norm = math.sqrt(w*w + qx*qx + qy*qy + qz*qz)
    return (w/norm, qx/norm, qy/norm, qz/norm)

def find_project_root_and_data(start_path: Path, max_depth: int = 6) -> tuple[Path, Path]:
    """start_path から親へ遡り、pam_force_map.csv を含む data/ を探す（tasks/data/）。見つからなければ FileNotFoundError。"""
    current = start_path.resolve()
    for _ in range(max_depth):
        candidate_data = current / "data"
        if candidate_data.exists() and candidate_data.is_dir():
             if (candidate_data / "pam_force_map.csv").exists():
                 return current, candidate_data
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError(
        f"Could not find 'data' directory containing 'pam_force_map.csv' by traversing up from {start_path}. "
        "Please ensure the 'data' folder exists in the project root."
    )

# assets/（USD）は data/ と同じ階層
try:
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT, DATA_DIR = find_project_root_and_data(BASE_DIR)
    ASSETS_DIR = PROJECT_ROOT / "assets"
    if not ASSETS_DIR.exists():
        print(f"[Warning] assets dir not found at {ASSETS_DIR}, trying to rely on defaults or manual set.")

except NameError:
    # __file__ がないとき（対話実行）はカレントディレクトリから
    PROJECT_ROOT, DATA_DIR = find_project_root_and_data(Path.cwd())
    ASSETS_DIR = PROJECT_ROOT / "assets"

print(f"[INFO] Asset Configuration:")
print(f"  - Project Root: {PROJECT_ROOT}")
print(f"  - Data Dir    : {DATA_DIR}")
print(f"  - Assets Dir  : {ASSETS_DIR}")

ROBOT_USD  = str(ASSETS_DIR / "porcaro.usd")
DRUM_USD   = str(ASSETS_DIR / "sneadrum.usd")

# 力マップが無ければ止める（黙って別の値で動かさない）
FORCE_MAP_CSV_PATH = str(DATA_DIR / "pam_force_map.csv")
if not os.path.exists(FORCE_MAP_CSV_PATH):
    raise FileNotFoundError(f"[CRITICAL] Force Map CSV NOT found at: {FORCE_MAP_CSV_PATH}")
FORCE_MAP_CSV = FORCE_MAP_CSV_PATH

H0_MAP_CSV_PATH    = str(DATA_DIR / "pam_force_0_map.csv")
if not os.path.exists(H0_MAP_CSV_PATH):
    print(f"[Warning] H0 Map CSV NOT found at: {H0_MAP_CSV_PATH}. Using None.")
    H0_MAP_CSV = None
else:
    H0_MAP_CSV = H0_MAP_CSV_PATH


ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",  # 環境の cfg で上書きする
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,  # 20 mm。大きくすると接触が早く始まり、打撃が柔らかくなる
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.11348, -0.34041, 0.0),
        rot=quat_from_euler_zyx(yaw_deg=30, pitch_deg=0, roll_deg=0),
        joint_pos={
            ".*Base_link_Wrist_joint": WRIST_J0,
            ".*Hand_link_Grip_joint":  GRIP_J0,
        },
    ),
    actuators={
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*Base_link_Wrist_joint"],
            stiffness=0.0,
            damping=0.1,
            effort_limit_sim=500.0,
            friction=0.001,
        ),
        "grip": ImplicitActuatorCfg(
            joint_names_expr=[".*Hand_link_Grip_joint"],
            stiffness=0.0,
            damping=0.001,
            effort_limit_sim=500.0,
            friction=0.00001,
        ),
    },
)

DRUM_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Drum",  # 環境の cfg で上書きする
    spawn=sim_utils.UsdFileCfg(
        usd_path=DRUM_USD,
        mass_props=MassPropertiesCfg(mass=1.0e2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,  # 20 mm（ロボット側と同じ）
            rest_offset=0.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.74573, 0.04029, 0.0),
        rot=quat_from_euler_zyx(yaw_deg=0, pitch_deg=0, roll_deg=0),
    ),
)
