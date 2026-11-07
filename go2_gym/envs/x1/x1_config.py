"""X1 robot assets, reference poses, and PD controller for the shared task."""

from go2_gym.envs.go2.go2_config import config_go2


def config_x1(cfg):
    # The source X1 and Go2 configs share the same terrain/randomization defaults.
    config_go2(cfg)
    cfg.init_state.default_joint_angles = {
        "joint_front_left_abad": 0.1,
        "joint_front_left_thigh_pitch": 0.7,
        "joint_front_left_calf_pitch": -1.5,
        "joint_front_right_abad": -0.1,
        "joint_front_right_thigh_pitch": 0.7,
        "joint_front_right_calf_pitch": -1.5,
        "joint_rear_left_abad": 0.1,
        "joint_rear_left_thigh_pitch": -0.7,
        "joint_rear_left_calf_pitch": 1.5,
        "joint_rear_right_abad": -0.1,
        "joint_rear_right_thigh_pitch": -0.7,
        "joint_rear_right_calf_pitch": 1.5,
    }
    cfg.control.control_type = "P"
    cfg.control.stiffness = {"joint": 30.0}
    cfg.control.damping = {"joint": 1.2}
    cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/dobot_quad_v2/urdf/dobot_quad.urdf"
    cfg.asset.terminate_after_contacts_on = ["link_trunk"]
    cfg.asset.flip_visual_attachments = False


def config_x1_go2bend(cfg):
    config_x1(cfg)
    for side in ("left", "right"):
        for joint in ("thigh_pitch", "calf_pitch"):
            cfg.init_state.default_joint_angles[f"joint_rear_{side}_{joint}"] = (
                cfg.init_state.default_joint_angles[f"joint_front_{side}_{joint}"]
            )
