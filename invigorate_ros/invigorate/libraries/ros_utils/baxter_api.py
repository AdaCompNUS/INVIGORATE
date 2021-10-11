import warnings
import numpy as np
from geometry_msgs.msg import (
        PoseStamped,
        Pose,
        Point,
        Quaternion,
    )
from std_msgs.msg import Header
import rospy
import baxter_interface
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from baxter_interface import CHECK_VERSION

H_OFFSET = 0.025

def init_baxter_robot():
    rs = baxter_interface.RobotEnable()
    rs.enable()
    return rs

def ik_solve(limb, pos, orient):
    # ~ rospy.init_node("rsdk_ik_service_client")
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    print "iksvc: ", iksvc
    print "ikreq: ", ikreq
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        str(limb): PoseStamped(header=hdr,
                               pose=Pose(position=pos, orientation=orient))}

    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    if (resp.isValid[0]):
        print("SUCCESS - Valid Joint Solution Found:")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print limb_joints
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")
    return -1

def move_limb_to_point(point, ori=None, limb='left', vel=1.0):
    limbhandle = baxter_interface.Limb(limb)
    limbhandle.set_joint_position_speed(vel)
    loc = Point(point[0], point[1], point[2])
    if ori is None:
        ori = [0, 1, 0, 0]
    ang = Quaternion(ori[0], ori[1], ori[2], ori[3])
    limbjoints = ik_solve(limb, loc, ang)
    limbhandle.move_to_joint_positions(limbjoints)

def move_baxter_to_grasp(robotgpoint, robotori, robotgvec, baxter_limb):
    # pre-grasp position
    robotprepoint = robotgpoint - 0.2 * robotgvec
    # baxter movement
    robotgpoint = robotgpoint - np.array([0, 0, H_OFFSET])
    move_limb_to_point(robotprepoint, robotori, limb=baxter_limb, vel=1.0)
    move_limb_to_point(robotgpoint, robotori, limb=baxter_limb, vel=1.0)

def move_limb_to_initial(initial=None, limb='left', vel=1.0):
    '''
    poselist: ['e0','e1','s0','s1','w0','w1','w2']
    default initial pose:
    [-0.9587379924283836, 1.8453788878261528, 0.3351748021529629, -1.373679795551388,
    0.14841264122791378, 1.1685098651717138, 0.03259709174256504]
    '''
    if initial is None:
        initial = {
            'left_e0': -0.9587379924283836,
            'left_e1': 1.8453788878261528,
            'left_s0': 0.3351748021529629,
            'left_s1': -1.373679795551388,
            'left_w0': 0.14841264122791378,
            'left_w1': 1.1685098651717138,
            'left_w2': 0.03259709174256504
        }

    limbhandle = baxter_interface.Limb(limb)
    limbhandle.set_joint_position_speed(vel)
    limbhandle.move_to_joint_positions(initial)

def move_limb_to_neutral(limb='left', vel=1.0):
    limbhandle = baxter_interface.Limb(limb)
    limbhandle.set_joint_position_speed(vel)
    limbhandle.move_to_neutral()

def grasp_and_put_thing_down(putposition, limb='left', vel=1.0):
    left_arm = baxter_interface.Limb('left')
    endpoints = left_arm.endpoint_pose()
    curpos = endpoints['position']
    curpos = np.array([curpos.x, curpos.y, curpos.z])
    curpos = curpos + np.array([0, 0, 0.3])
    gripper = baxter_interface.Gripper(limb, CHECK_VERSION)
    rospy.sleep(0.5)
    gripper.close()
    rospy.sleep(0.5)
    preputpos = putposition + np.array([0, 0, 0.2])
    move_limb_to_point(point=curpos, limb=limb, vel=vel)
    move_limb_to_point(point=preputpos, limb=limb, vel=vel)
    # move_limb_to_point(point = putposition, limb=limb, vel = vel)
    gripper.open()
    rospy.sleep(0.5)
    move_limb_to_point(point=preputpos, limb=limb, vel=vel)