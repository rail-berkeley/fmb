import time
import gym
import robot_infra
import numpy as np
env = gym.make('Franka-FMB-v0')
o = env.reset()

from robot_infra.spacemouse.spacemouse_expert import SpaceMouseExpert

spacemouse = SpaceMouseExpert()


def spacemouse_subtraj(env, o, expert, type_traj, classifier=None):
    traj = dict(
        observations=[],
        actions=[],
    )
    t = 0
    grasp = env.currgrip
    env.compliance_mode()
    print(type_traj)
    if type_traj == 'close':
        print(f"RESET MODE")
    elif type_traj == 'grasp':
        env.grasp_mode()
    elif type_traj == 'insert':
        env.insertion_mode()
        env.compliance_mode()
    
    while True:
        
        ## Get action from spacemouse
        controller_a, button = expert.get_action()
        left = button[0]
        right = button[1]
        a = np.zeros((7,))
        a[:6] = controller_a
        a *= env.action_space.high
        
        ## Scaling for regrasp
        if 'regrasp' in type_traj or 'place_on_fixture' in type_traj:
            a[3:5] *= 2
        ## Throw out action and timestep if actino is close to zero
        if np.linalg.norm(a) < 0.001 and not right and not left:
            continue

        ## Get grasp action
        if right:
           grasp = not grasp
           time.sleep(0.2)
        a[-1] = 1 if grasp else 0  ## GRIPPER

        ## Append action and observation to trajectory
        traj['observations'].append(o)
        traj['actions'].append(a)

        ## Step
        o, _, _, _ = env.step(a)
        if classifier:
            print(f"REWARD: {classifier(o)}")
        ## Print timestep
        t += 1
        # sys.stdout.write(f"Timestep {t}" )
        # sys.stdout.flush()

        if type_traj is not 'close':
            print(f"Timestep: {t}", end='\r')
 
        ## Finish if button pressed
        if left:
            if 'grasp' in type_traj:
                filler_a = np.zeros((7,))
                filler_a[-1] = 1
                traj['observations'].append(o)
                traj['actions'].append(filler_a)
                o, _, _, _ = env.step(filler_a)
                for i in range(15):
                    random_a = np.random.uniform(-0.2, 0.2,(7,)) * env.action_space.high
                    random_a[-1] = 1
                    traj['observations'].append(o)
                    traj['actions'].append(filler_a)
                    o, _, _, _ = env.step(random_a)
                    print(f'Random Motion Timestep: {i+1}', end='\r')
            elif type_traj == 'place_on_fixture':
                traj['observations'].append(o)
                a = np.zeros(7)
                traj['actions'].append(a)
                o, _, _, _ = env.step(a)
            break
    env.freespace_mode()
    env.precision_mode()
    return o, traj

spacemouse_subtraj(env, o, spacemouse, 'grasp')