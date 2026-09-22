import os.path
import time

import numpy as np

import mplib
from mplib import Pose
from mplib.collision_detection import fcl

# Import planning strategies with perturbation fallback
from lilo_vla.reaching.motion_planning.planning_strategies import (
    plan_pose_with_perturbation_fallback,
)

class MPLibPlanner(object):
    def __init__(self, base_pose, move_group="panda_hand",
                 arm_dim=7, time_step=0.01, scene_resolution=0.01):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        urdf = f"{root_dir}/robot_models/panda.urdf"
        srdf = f"{root_dir}/robot_models/panda.srdf"

        self.mv_link_to_ctrl = np.array([
            [-4.92624354e-04, -9.99999879e-01, -9.36859320e-12, -3.44339487e-12],
            [ 9.99999879e-01, -4.92624354e-04, -1.02778751e-13, -4.44262682e-14],
            [ 9.81549444e-14, -9.36864536e-12,  1.00000000e+00, -9.65000000e-02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ])
        self.ctrl_to_mv_link = np.linalg.inv(self.mv_link_to_ctrl)

        # todo make this flexible

        self.arm_dim = arm_dim
        self.time_step = time_step
        self.scene_reso = scene_resolution

        self.pts_names = []
        planner = mplib.Planner(
            urdf=urdf,
            srdf=srdf,
            move_group=move_group,
        )
        self.move_group = move_group
        self.planner = planner

        self.base_pose = base_pose
        self.planner.set_base_pose(Pose(base_pose))

    def clear_scene(self):
        for name in self.pts_names:
            self.remove_pts(name)

    def remove_pts(self, name):
        if name in self.pts_names:
            self.pts_names.remove(name)
        return self.planner.remove_point_cloud(name)

    def update_scene(self, scene_pts, name):
        self.pts_names.append(name)
        self.planner.update_point_cloud(scene_pts, name=name, resolution=self.scene_reso)

    def attach_obj(self, grasp_pose, aabb):
        idx = self.planner.user_link_names.index(self.move_group)
        robot = self.planner.robot
        # CRITICAL FIX: aabb from get_object_bounding_box() is half-extents,
        # but MPlib's update_attached_box expects FULL side lengths
        # Convert half-extents to full side lengths
        # aabb[2] = aabb[2] * .5
        full_size = np.array(aabb) * 2.0
        self.planner.update_attached_box(full_size, pose=Pose(grasp_pose),
                                         art_name=robot.name, link_id=idx)

    def detach_obj(self, also_remove=True):
        self.planner.detach_object(also_remove=also_remove)

    def plan_to_pose(self, q0, tar_ctrl_pose, verbose=False):
        if len(q0) == self.arm_dim:
            q0 = np.concatenate([q0, np.zeros(2)], axis=0)

        # # Convert target pose from controller to move_group link
        # # tar_ctrl_pose is controller target in WORLD frame
        # # Planner needs move_group link target in WORLD frame
        # # Since move_group is 9.65cm ABOVE controller, transform accordingly
        # mv_link_pose = Pose(tar_ctrl_pose @ self.ctrl_to_mv_link)

        # ========================================
        # NEW: Planning pipeline with perturbation fallback
        # Pipeline tries: Original pose (3 strategies) → Perturbed poses (up to 50 attempts)
        # Fixed perturbations (5 attempts): +Z (lift 2cm), +X, -X, +X+rot, +X-rot, then random
        # ========================================
        res = plan_pose_with_perturbation_fallback(
            self, q0, tar_ctrl_pose,
            max_perturbations=50,
            use_fixed_perturbations=True,  # Try 5 fixed perturbations before random
            xy_range=0.04,      # ±4cm XY shift
            z_range=0.02,       # ±2cm Z shift
            ori_range=15.0,     # ±15° rotation around z-axis
            time_step=self.time_step,
            mv_link_to_ctrl=self.mv_link_to_ctrl,
            planning_time=10.0,
            verbose=verbose
        )

        # ========================================
        # OLD CODE (COMMENTED OUT FOR COMPARISON)
        # ========================================
        # mv_link_pose = tar_ctrl_pose @ self.mv_link_to_ctrl
        # mv_link_pose = Pose(mv_link_pose)
        #
        # # STRATEGY 1: Try old simple retry with increased planning budget first
        # if verbose:
        #     print(f"   Attempting OLD CODE: Simple retry with increased planning budget...")
        # res = None
        # for attempt in range(3):
        #     res = self.planner.plan_pose(
        #         mv_link_pose, q0,
        #         time_step=self.time_step,
        #         wrt_world=True,
        #         planning_time=10.0  # OPTIMIZED: Extended from 5.0s for better RRT paths
        #     )
        #
        #     if res["status"] == "Success":
        #         res["score"] = 1.
        #         res["cartesian"] = self.convert_joint_to_ctrl_poses(res["position"])
        #         if verbose:
        #             print(f"   ✅ Planning succeeded with OLD CODE (simple retry, attempt {attempt + 1}/3)")
        #         break
        #     else:
        #         if verbose:
        #             print(f"   MPLib planning attempt {attempt + 1}/3 failed: {res['status']}")
        #
        # if res is None or res["status"] != "Success":
        #     if verbose:
        #         print(f"   ❌ OLD CODE failed after 3 attempts")
        #
        #     # STRATEGY 2: Try Strategy B (adaptive subdivision)
        #     if verbose:
        #         print(f"   Attempting Strategy B: Adaptive subdivision...")
        #     res = plan_with_adaptive_subdivision(
        #         self, q0, tar_ctrl_pose,
        #         max_depth=3,
        #         time_step=self.time_step,
        #         mv_link_to_ctrl=self.mv_link_to_ctrl,
        #         planning_time=10.0,
        #         verbose=verbose
        #     )
        #
        #     if res["status"] == "Success":
        #         if verbose:
        #             print(f"   ✅ Planning succeeded with STRATEGY B (adaptive subdivision)")
        #     else:
        #         if verbose:
        #             print(f"   ❌ Strategy B failed")
        #
        #         # STRATEGY 3: Try Strategy A (fixed waypoints)
        #         if verbose:
        #             print(f"   Attempting Strategy A: Fixed waypoints...")
        #         res = plan_with_fixed_waypoints(
        #             self, q0, tar_ctrl_pose,
        #             num_waypoints=2,
        #             time_step=self.time_step,
        #             mv_link_to_ctrl=self.mv_link_to_ctrl,
        #             planning_time=10.0,
        #             verbose=verbose
        #         )
        #
        #         if res["status"] == "Success":
        #             if verbose:
        #                 print(f"   ✅ Planning succeeded with STRATEGY A (fixed waypoints)")
        #         else:
        #             if verbose:
        #                 print(f"   ❌ All strategies failed (OLD CODE, Strategy B, Strategy A)")
        #             res = {"status": "Failed", "score": 0., "position": [q0[:self.arm_dim]], "cartesian": []}

        # v1 code   
        # res = self.planner.plan_pose(mv_link_pose, q0, time_step=self.time_step,
        #                              wrt_world=True)

        # if res["status"] == "Success":
        #     res["score"] = 1.
        #     res["cartesian"] = self.convert_joint_to_ctrl_poses(res["position"])
        # else:
        #     exit(1)
            # res["score"] = 0.
            # cur_ctrl_pose = self.convert_joint_to_ctrl_poses(q0[None])
            # interp_ctrl_poses = np.concatenate([cur_ctrl_pose, tar_ctrl_pose[None]], axis=0)
            # dist = np.linalg.norm(interp_ctrl_poses[0, :3, 3] - interp_ctrl_poses[1, :3, 3])
            # interp_steps = int(dist / 0.05)
            # res["cartesian"] = interpolate_object_trajectory(interp_ctrl_poses, interp_steps)

            # ik_status, q1 = self.planner.IK(self.planner._transform_goal_to_wrt_base(mv_link_pose),
            #                                 start_qpos=q0,
            #                                 n_init_qpos=10,
            #                                 return_closest=True)
            # if ik_status == "Success":
            #     res["position"] = np.array([q0, q1])
            # else:
            #     res["position"] = np.array([q0, q0])

        res["position"] = np.array(res["position"])[:, :self.arm_dim]
        # print("plan traj: ", res["position"].shape)

        return res

    def plan_traj(self, q0, tar_ctrl_poses):
        if len(q0) == self.arm_dim:
            q0 = np.concatenate([q0, np.zeros(2)], axis=0)

        # if self.link7_to_ctrl is not None:
        tar_mv_link_poses = tar_ctrl_poses @ self.mv_link_to_ctrl[None]

        cur_q = q0
        n = len(tar_mv_link_poses)
        qposes = []
        scores = []

        for i in range(n):
            tar_mv_pose = self.planner._transform_goal_to_wrt_base(Pose(tar_mv_link_poses[i]))
            ik_status, goal_qpos = self.planner.IK(tar_mv_pose, cur_q,
                                                   n_init_qpos=10,
                                                   return_closest=True)
            if ik_status != "Success":
                scores.append(0)
                qposes.append(cur_q)
            else:
                scores.append(1)
                qposes.append(goal_qpos)
                cur_q = goal_qpos

        qposes = np.array(qposes)

        res = self.planner.plan_qpos(qposes, q0, time_step=self.time_step)

        if res["status"] == "Success":
            score_plan = 1.
            if len(res["position"]) == 0:
                res["position"] = qposes
        else:
            score_plan = 0.
            res["position"] = qposes

        res["score"] = np.mean(scores) * (score_plan)

        # print("plan traj: ", res["position"].shape)

        res["position"] = np.array(res["position"])[:, :self.arm_dim]
        res["cartesian"] = tar_ctrl_poses

        return res

    # def convert_joint_to_ctrl_poses(self, joint_traj):
    #     ctrl_poses = []
    #     for q in joint_traj:
    #         if len(q) == 7:
    #             q = np.concatenate([np.array(q), np.zeros(2)], axis=0)

    #         self.set_qpos(q)
    #         mv2world = self.get_link_pose(wrt_world=True)
    #         # Convert from move_group link frame to controller frame
    #         # FIXED: Was using ctrl_to_mv_link (wrong direction), now uses mv_link_to_ctrl (correct)
    #         ctrl2world = mv2world @ self.mv_link_to_ctrl
    #         ctrl_poses.append(ctrl2world)
    #     return np.array(ctrl_poses)

    def convert_joint_to_ctrl_poses(self, joint_traj):
        ctrl_poses = []
        for q in joint_traj:
            if len(q) == 7:
                q = np.concatenate([np.array(q), np.zeros(2)], axis=0)

            self.set_qpos(q)
            mv2world = self.get_link_pose(wrt_world=True)
            ctrl2world = mv2world @ self.ctrl_to_mv_link
            ctrl_poses.append(ctrl2world)
        return np.array(ctrl_poses)

    def set_qpos(self, q0):
        self.planner.robot.set_qpos(q0, True)

    def get_link_pose(self, wrt_world=True):
        idx = self.planner.user_link_names.index(self.move_group)
        link_pose = np.array(self.planner.pinocchio_model.get_link_pose(idx).to_transformation_matrix())

        if wrt_world:
            link_pose = self.base_pose @ link_pose
        return link_pose


def compute_rel_T(env, n=10):
    from lilo_vla.reaching.motion_planning.action_utils import pose_traj_to_action, get_controller_robot_pose, interpolate

    robot_model = env.robots[0].robot_model
    base_pos = env.sim.data.get_body_xpos(robot_model.root_body)

    print("base pos: ", base_pos)

    base_pose = np.eye(4)
    base_pose[:3, 3] = np.array(base_pos)
    planner = MPLibPlanner(base_pose)

    for i in range(n):
        obs = env.reset()

        curr_pos, curr_rot = get_controller_robot_pose(env, "right")
        ctrl_pose = np.eye(4)
        ctrl_pose[:3, :3] = curr_rot
        ctrl_pose[:3, 3] = curr_pos

        q0 = obs["robot0_joint_pos"].tolist() + [0, 0]
        planner.set_qpos(q0)
        link7_pose = planner.get_link_pose()
        link7_to_ctrl = np.linalg.inv(ctrl_pose) @ link7_pose
        print()
        print(link7_to_ctrl)
        print()

