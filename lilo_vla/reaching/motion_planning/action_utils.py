import numpy as np
import robosuite.utils.transform_utils as T
import robosuite
import math

def get_controller_robot_pose(env, arm_name):
    """
    Helper function to get controller robot pose from current state of environment.
    """
    # For LIBERO environments, eef_site_id is typically just an integer, not a dictionary
    # Let's handle both cases
	# eef_site_id = env.robots[0].eef_site_id  # Site ID 7 = "gripper0_grip_site"
    try:
        # Try as dictionary first (some robosuite versions)
        if hasattr(env.robots[0], 'eef_site_id') and isinstance(env.robots[0].eef_site_id, dict):
            eef_site_id = env.robots[0].eef_site_id[arm_name]
        else:
            # Try as direct integer (LIBERO environments)
            eef_site_id = env.robots[0].eef_site_id
    except (AttributeError, TypeError, KeyError):
        # Fallback: try to find the site by name
        try:
            eef_site_id = env.sim.model.site_name2id('robot0_grip_site')
        except:
            # Last resort: use a common default
            eef_site_id = env.robots[0].eef_site_id if hasattr(env.robots[0], 'eef_site_id') else 0

    curr_pos = np.array(env.sim.data.site_xpos[eef_site_id])
    curr_rot = np.array(env.sim.data.site_xmat[eef_site_id].reshape([3, 3]))
    return curr_pos, curr_rot

def quat2axisangle(quat):
	"""
	Converts (x, y, z, w) quaternion to axis-angle format.
	Returns a unit vector direction and an angle.

	NOTE: this differs from robosuite's function because it returns
		  both axis and angle, not axis * angle.
	"""

	# conversion from axis-angle to quaternion:
	#   qw = cos(theta / 2); qx, qy, qz = u * sin(theta / 2)

	# normalize qx, qy, qz by sqrt(qx^2 + qy^2 + qz^2) = sqrt(1 - qw^2)
	# to extract the unit vector

	# clipping for scalar with if-else is orders of magnitude faster than numpy
	if quat[3] > 1.:
		quat[3] = 1.
	elif quat[3] < -1.:
		quat[3] = -1.

	den = np.sqrt(1. - quat[3] * quat[3])
	if math.isclose(den, 0.):
		# This is (close to) a zero degree rotation, immediately return
		return np.zeros(3), 0.

	return quat[:3] / den, 2. * math.acos(quat[3])

def axisangle2quat(axis, angle):
	"""
	Converts axis-angle to (x, y, z, w) quat.

	NOTE: this differs from robosuite's function because it accepts
		  both axis and angle as arguments, not axis * angle.
	"""

	# handle zero-rotation case
	if math.isclose(angle, 0.):
		return np.array([0., 0., 0., 1.])

	# make sure that axis is a unit vector
	assert math.isclose(np.linalg.norm(axis), 1., abs_tol=1e-3)

	q = np.zeros(4)
	q[3] = np.cos(angle / 2.)
	q[:3] = axis * np.sin(angle / 2.)
	return q

def poses_to_action(start_pos, target_pos, start_rot=None, target_rot=None, max_dpos=None, max_drot=None):
	"""
	Takes a starting eef pose and target controller pose and returns a normalized action that
	corresponds to the desired controller target.

	NOTE: assumes robosuite v1.2, for the convention used to translate a delta rotation action into absolute
		  rotation target
	"""
	delta_position = target_pos - start_pos
	delta_position = np.clip(delta_position / max_dpos, -1., 1.)
	if target_rot is None:
		return delta_position

	# version check for robosuite - must be v1.2, so that we're using the correct controller convention
	# assert (robosuite.__version__.split(".")[0] == "1")
	# assert (robosuite.__version__.split(".")[1] == "2")

	# use the OSC controller's convention for delta rotation
	delta_rot_mat = target_rot.dot(start_rot.T)
	delta_quat = T.mat2quat(delta_rot_mat)
	delta_rotation = T.quat2axisangle(delta_quat)
	delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
	return np.concatenate([delta_position, delta_rotation])

def interpolate_poses(pos1, rot1, pos2, rot2, num_steps=None, step_size=None, perturb=False):
	"""
	Linear interpolation between two poses.

	Args:
		pos1 (np.array): np array of shape (3,) for first position
		rot1 (np.array): np array of shape (3, 3) for first rotation
		pos2 (np.array): np array of shape (3,) for second position
		rot2 (np.array): np array of shape (3, 3) for second rotation
		num_steps (int): if provided, specifies the number of desired interpolated points (not excluding
			the start and end points). Passing 0 corresponds to no interpolation, and passing None
			means that @step_size must be provided to determine the number of interpolated points.
		step_size (float): if provided, will be used to infer the number of steps, by taking the norm
			of the delta position vector, and dividing it by the step size
		perturb (bool): if True, randomly move all the interpolated position points in a uniform, non-overlapping grid.

	Returns:
		pos_steps (np.array): array of shape (N + 2, 3) corresponding to the interpolated position path, where N is @num_steps
		rot_steps (np.array): array of shape (N + 2, 3, 3) corresponding to the interpolated rotation path, where N is @num_steps
		num_steps (int): the number of interpolated points (N) in the path
	"""
	assert step_size is None or num_steps is None

	if num_steps == 0:
		# skip interpolation
		return np.concatenate([pos1[None], pos2[None]], axis=0), np.concatenate([rot1[None], rot2[None]],
																				axis=0), num_steps

	delta_pos = pos2 - pos1
	if num_steps is None:
		assert np.linalg.norm(delta_pos) > 0
		num_steps = math.ceil(np.linalg.norm(delta_pos) / step_size)

	num_steps += 1  # include starting pose
	assert num_steps >= 2

	# linear interpolation of positions
	pos_step_size = delta_pos / num_steps
	grid = np.arange(num_steps).astype(np.float64)
	if perturb:
		# move the interpolation grid points by up to a half-size forward or backward
		perturbations = np.random.uniform(
			low=-0.5,
			high=0.5,
			size=(num_steps - 2,),
		)
		grid[1:-1] += perturbations
	pos_steps = np.array([pos1 + grid[i] * pos_step_size for i in range(num_steps)])

	# add in endpoint
	pos_steps = np.concatenate([pos_steps, pos2[None]], axis=0)

	# interpolate the rotations too
	rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps, axis_angle=True)

	return pos_steps, rot_steps, num_steps - 1

def quat_slerp(q1, q2, tau):
	"""
	Adapted from robosuite.
	"""
	if tau == 0.0:
		return q1
	elif tau == 1.0:
		return q2
	d = np.dot(q1, q2)
	if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.:
		return q1
	if d < 0.0:
		# invert rotation
		d = -d
		q2 *= -1.0
	angle = math.acos(np.clip(d, -1, 1))
	if abs(angle) < np.finfo(float).eps * 4.:
		return q1
	isin = 1.0 / math.sin(angle)
	q1 = q1 * math.sin((1.0 - tau) * angle) * isin
	q2 = q2 * math.sin(tau * angle) * isin
	q1 = q1 + q2
	return q1

def interpolate_rotations(R1, R2, num_steps, axis_angle=True):
	"""
	Interpolate between 2 rotation matrices. If @axis_angle, interpolate the axis-angle representation
	of the delta rotation, else, use slerp.

	NOTE: I have verified empirically that both methods are essentially equivalent, so pick your favorite.
	"""
	if axis_angle:
		# delta rotation expressed as axis-angle
		delta_rot_mat = R2.dot(R1.T)
		delta_quat = T.mat2quat(delta_rot_mat)
		delta_axis, delta_angle = quat2axisangle(delta_quat)

		# fix the axis, and chunk the angle up into steps
		rot_step_size = delta_angle / num_steps

		# convert into delta rotation matrices, and then convert to absolute rotations
		if delta_angle < 0.05:
			# small angle - don't bother with interpolation
			rot_steps = np.array([R2 for _ in range(num_steps)])
		else:
			delta_rot_steps = [T.quat2mat(axisangle2quat(delta_axis, i * rot_step_size)) for i in range(num_steps)]
			rot_steps = np.array([delta_rot_steps[i].dot(R1) for i in range(num_steps)])
	else:
		q1 = T.mat2quat(R1)
		q2 = T.mat2quat(R2)
		rot_steps = np.array([T.quat2mat(quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])

	# add in endpoint
	rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

	return rot_steps

def pose_traj_to_action(env, target_pos, target_rot=None, pos_interp_th=None, rot_interp_th=None,
						arm_name="right", velocity_factor=0.9):
	"""
	Take a controller target pose and return a normalized action (usually a normalized
	delta pose action) that corresponds to setting the controller target pose to the
	input value. To compute this, the current eef pose will be read from the env.

	Args:
		target_pos (np.array): target position
		target_rot (np.array): target rotation
		velocity_factor (float): Factor to scale controller limits (0.0-1.0, default: 0.9)

	Returns:
		(np.array) pose action
	"""
	# Handle different controller structures (LIBERO vs standard robosuite)
	try:
		# Try composite controller first (multi-arm setups)
		if hasattr(env.robots[0], 'composite_controller'):
			controllers = env.robots[0].composite_controller.part_controllers
			arm_controller = controllers[arm_name]
		else:
			# Single arm setup - controller is directly accessible
			arm_controller = env.robots[0].controller
	except (AttributeError, KeyError):
		# Fallback to direct controller access
		arm_controller = env.robots[0].controller

	# Get controller limits
	try:
		max_dpos = arm_controller.output_max[0]
		max_drot = None if target_rot is None else arm_controller.output_max[3]
	except (AttributeError, IndexError):
		# Fallback to reasonable defaults if controller limits not available
		max_dpos = 0.1  # 10cm max per step
		max_drot = 0.5  # ~28 degrees max per step

	pos_interp_th = max_dpos * velocity_factor
	rot_interp_th = max_drot * velocity_factor

	curr_pos, curr_rot = get_controller_robot_pose(env, arm_name=arm_name)

	pos_diff = np.linalg.norm(target_pos - curr_pos)
	delta_rot_mat = target_rot @ (curr_rot.T)
	delta_quat = T.mat2quat(delta_rot_mat)
	ang_diff = T.quat2axisangle(delta_quat)
	ang_diff = np.linalg.norm(ang_diff)

	path_len = 2

	if (pos_interp_th is not None and pos_diff > pos_interp_th) or\
			(rot_interp_th is not None and ang_diff > rot_interp_th):
		pos_steps = int(pos_diff / pos_interp_th)
		if pos_steps * pos_interp_th < pos_diff:
			pos_steps += 1
		rot_steps = int(ang_diff / rot_interp_th)
		if rot_steps * rot_interp_th < ang_diff:
			rot_steps += 1

		pos_path, rot_path, num_steps = interpolate_poses(
			pos1=curr_pos,
			rot1=curr_rot,
			pos2=target_pos,
			rot2=target_rot,
			num_steps=max(rot_steps, pos_steps)
		)
		target_rot = rot_path[1]
		target_pos = pos_path[1]

		path_len = len(rot_path)

	action_pos = poses_to_action(
		start_pos=curr_pos,
		target_pos=target_pos,
		start_rot=None if target_rot is None else curr_rot,
		target_rot=target_rot,
		max_dpos=max_dpos,
		max_drot=max_drot
	)

	target_pose = np.eye(4)
	target_pose[:3, :3] = target_rot
	target_pose[:3, 3] = target_pos

	curr_pos, curr_rot = get_controller_robot_pose(env, arm_name=arm_name)

	new_pos_diff = np.linalg.norm(target_pos - curr_pos)
	delta_rot_mat = target_rot @ (curr_rot.T)
	delta_quat = T.mat2quat(delta_rot_mat)
	new_ang_diff = T.quat2axisangle(delta_quat)
	new_ang_diff = np.linalg.norm(new_ang_diff)

	ret = {"action": action_pos, "target_pose": target_pose,
		   "pos_err": new_pos_diff, "rot_err": new_ang_diff,
		   "path_len": path_len
		   }

	return ret

def interpolate(env, arm_name, target_pos, target_rot, velocity_factor=0.8):
	# Handle different controller structures (LIBERO vs standard robosuite)
	try:
		# Try composite controller first (multi-arm setups)
		if hasattr(env.robots[0], 'composite_controller'):
			controllers = env.robots[0].composite_controller.part_controllers
			arm_controller = controllers[arm_name]
		else:
			# Single arm setup - controller is directly accessible
			arm_controller = env.robots[0].controller
	except (AttributeError, KeyError):
		# Fallback to direct controller access
		arm_controller = env.robots[0].controller

	# Get controller limits
	try:
		max_dpos = arm_controller.output_max[0]
		max_drot = None if target_rot is None else arm_controller.output_max[3]
	except (AttributeError, IndexError):
		# Fallback to reasonable defaults if controller limits not available
		max_dpos = 0.1  # 10cm max per step
		max_drot = 0.5  # ~28 degrees max per step

	pos_interp_th = max_dpos * velocity_factor
	rot_interp_th = max_drot * velocity_factor

	curr_pos, curr_rot = get_controller_robot_pose(env, arm_name=arm_name)

	pos_diff = np.linalg.norm(target_pos - curr_pos)
	delta_rot_mat = target_rot @ (curr_rot.T)
	delta_quat = T.mat2quat(delta_rot_mat)
	ang_diff = T.quat2axisangle(delta_quat)
	ang_diff = np.linalg.norm(ang_diff)

	pos_steps = int(pos_diff / pos_interp_th)
	if pos_steps * pos_interp_th < pos_diff:
		pos_steps += 1
	rot_steps = int(ang_diff / rot_interp_th)
	if rot_steps * rot_interp_th < ang_diff:
		rot_steps += 1

	pos_path, rot_path, num_steps = interpolate_poses(
		pos1=curr_pos,
		rot1=curr_rot,
		pos2=target_pos,
		rot2=target_rot,
		num_steps=max(rot_steps, pos_steps)
	)

	return pos_path, rot_path