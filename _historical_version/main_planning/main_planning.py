import numpy as np
import pickle
import control as ct
from waypoint_plan import plan_waypoints, Node
from time_plan import plan_times
from vis_waypoints import vis_waypoints
from vis_time import vis_time
from bounding_box_plan import optimize_boundingbox
import matplotlib.pyplot as plt; plt.rcdefaults()
import plot_figures as pf


class SingleSwarm:
    def __init__(self, id, dynamics, total_num_agents, lqr_Q_R):
        self.id = id
        self.num_agents = total_num_agents[id]
        self.dynamics = dynamics
        self.lqr_Q_R = lqr_Q_R  #[Q,R] matrix
        self.B = cholesky_decomposition_lqr(dynamics, lqr_Q_R)
        # self.bounding_box_A = bounding_box_A
        # self.bounding_box_sigma = np.matmul(bounding_box_A, bounding_box_A.T)

def cholesky_decomposition_lqr(dynamics, lqr_Q_R = (np.eye(2), np.eye(2))):
    A = dynamics[0]
    B = dynamics[1]
    Q = lqr_Q_R[0]
    R = lqr_Q_R[1]
    _, P, _ = ct.lqr(A, B, Q, R)
    try:
        B_p = np.linalg.cholesky(P)
        return B_p
    except np.linalg.LinAlgError:
        print("Matrix P is not positive definite.")

def test_waypoint(time_stamps, swarm,boxes,num_segs=9):
    global tmax
    x0 = [-1., -1]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([-1.5, -1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([1.5, 1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, -1.5, -1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, 1.5, 1.5], dtype=np.float64))

    obs = []
    for wall in walls:
        if wall[0]==wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2]==wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1,1,-1,1])
        obs.append((A, wall))

    b1 = np.array([1, -0.7, 0.25, 0.5], dtype = np.float64)
    B1 = (A, b1)
    b2 = np.array([0, 0.9, 1, -0.5], dtype = np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype = np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype = np.float64)
    C = (A, C)
    # inside the wall
    D = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    D = (A, D)

    goal = [1,1]

    plots = [[[B1,], 'y'], [[B2,], 'r'], [[B3,], 'g'], [[C,], 'b'], [obs, 'k']]
    # tmax = 200.
    vmax = 1.

    notC = Node('negmu', info={'A':C[0], 'b':C[1]})
    B1 = Node('mu', info={'A':B1[0], 'b':B1[1]})
    B2 = Node('mu', info={'A':B2[0], 'b':B2[1]})
    B3 = Node('mu', info={'A':B3[0], 'b':B3[1]})
    alwaysD = Node('mu', info={'A':D[0], 'b':D[1]})

    phi_1 = Node('F', deps=[Node('A', deps=[B2,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_2 = Node('F', deps=[Node('A', deps=[B3,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_3 = Node('A', deps=[notC,], info={'int':[0,tmax]})
    phi_4 = Node('A', deps=[alwaysD, ], info={'int': [0, tmax]})
    spec = Node('and', deps=[phi_1, phi_2, phi_3, phi_4])

    x0s = [x0,]
    specs = [spec,]
    goals = [goal, ]
    PWL, margin, total_time = plan_waypoints(time_stamps, swarm,boxes, x0s, specs, bloat=0.05, size=0.11/2, num_segs=19, tmax=tmax, vmax=vmax, hard_goals=goals)

    return x0s, plots, PWL, margin, boxes, total_time

def test_time(initial_PWLs, swarm,boxes,num_segs=9):
    global tmax
    x0 = [-1., -1]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([-1.5, -1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([1.5, 1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, -1.5, -1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, 1.5, 1.5], dtype=np.float64))

    obs = []
    for wall in walls:
        if wall[0]==wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2]==wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1,1,-1,1])
        obs.append((A, wall))

    b1 = np.array([1, -0.7, 0.25, 0.5], dtype = np.float64)
    B1 = (A, b1)
    b2 = np.array([0, 0.9, 1, -0.5], dtype = np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype = np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype = np.float64)
    C = (A, C)
    # inside the wall
    D = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    D = (A, D)

    goal = [1,1]

    plots = [[[B1,], 'y'], [[B2,], 'r'], [[B3,], 'g'], [[C,], 'b'], [obs, 'k']]

    vmax = 1.

    notC = Node('negmu', info={'A':C[0], 'b':C[1]})
    B1 = Node('mu', info={'A':B1[0], 'b':B1[1]})
    B2 = Node('mu', info={'A':B2[0], 'b':B2[1]})
    B3 = Node('mu', info={'A':B3[0], 'b':B3[1]})
    alwaysD = Node('mu', info={'A':D[0], 'b':D[1]})

    phi_1 = Node('F', deps=[Node('A', deps=[B2,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_2 = Node('F', deps=[Node('A', deps=[B3,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_3 = Node('A', deps=[notC,], info={'int':[0,tmax]})
    phi_4 = Node('A', deps=[alwaysD, ], info={'int': [0, tmax]})
    spec = Node('and', deps=[phi_1, phi_2, phi_3, phi_4])

    x0s = [x0,]
    specs = [spec,]
    goals = [goal, ]
    # PWL = plan(swarm, x0s, specs, bloat=0.05, size=0.11/2, num_segs=9, tmax=tmax, vmax=vmax, hard_goals=goals)
    PWL, margin, running_time = plan_times(initial_PWLs, swarm,boxes, x0s, specs, bloat=0.05, size=0.11/2, num_segs=19, tmax=tmax, vmax=vmax, hard_goals=goals)

    return x0s, plots, PWL, margin, boxes, running_time

def task_environment():
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []
    wall_half_width = 0.05

    walls.append(np.array([-1.5, -1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([1.5, 1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, -1.5, -1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, 1.5, 1.5], dtype=np.float64))

    obs = []
    for wall in walls:
        if wall[0] == wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2] == wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1, 1, -1, 1])
        obs.append((A, wall))

    b1 = np.array([1, -0.7, 0.25, 0.5], dtype=np.float64)
    B1 = (A, b1)
    b2 = np.array([0, 0.9, 1, -0.5], dtype=np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype=np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float64)
    C = (A, C)
    # inside the wall
    D = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    D = (A, D)
    test_list = [B2, B3, C, D]
    plots = [[[B1, ], 'y'], [[B2, ], 'r'], [[B3, ], 'g'], [[C, ], 'b'], [obs, 'k']]

    return test_list, plots

if __name__ == "__main__":
    num_rounds = 20
    eta = 0

    tmax = 200

    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    dynamics = [A, B]
    total_num_agents = [2, 3, 4, 5]
    print(dynamics[1])
    print(total_num_agents[0])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    lqr_set = [Q, R]
    swarm = SingleSwarm(1, dynamics, total_num_agents, lqr_set)
    num_segs = 19
    test_list, _ = task_environment()
    # Note: Sigma of bounding box = box @ box.T
    initial_Box = np.array([[0.001, 0], [0, 0.001]])
    initial_Boxes = []
    for i in range(num_segs+1):
        initial_Boxes.append(initial_Box)

    initial_time_stamps = [0]
    for i in range(num_segs):
        temp = initial_time_stamps[-1] + 1
        initial_time_stamps.append(temp)

    total_time = []
    margins = []
    for num_round in range(num_rounds):
        given_time_stamps = []
        # num_round == 1
        if num_round == 0:
            given_time_stamps = initial_time_stamps
            Boxes = initial_Boxes
        else:
            with open('PWLs_output_bounding_Boxes.pkl', 'rb') as file:
                PWLs_loaded, margin_loaded, Box_loaded = pickle.load(file)
                print(margin_loaded)
            for PWL in PWLs_loaded:
                for i in range(num_segs + 1):
                    given_time_stamps.append(PWLs[0][i][1])

            Boxes_temp = Box_loaded[0]
            Boxes[0] = Boxes_temp[1][0]
            for j  in range(1, num_segs + 1):
                Boxes[j] = Boxes_temp[j][0]





        # results = vis_waypoints(test_waypoint, given_time_stamps, swarm, Boxes)
        running_time_1 = vis_waypoints(test_waypoint, given_time_stamps, swarm, Boxes)
        print(f"waypoint finish round {num_round}, next we optimize over time")
        #
        with open('PWLs_output_waypoints.pkl', 'rb') as file:
            PWLs_loaded, margin_loaded, Box_loaded = pickle.load(file)

        initial_PWLs = PWLs_loaded
        Boxes = Box_loaded
        # results = vis_time(test_time, initial_PWLs, swarm, Boxes)
        running_time_2 = vis_time(test_time, initial_PWLs, swarm, Boxes)

        print(f"time finish round {num_round}, next we optimize over bounding box")
        with open('PWLs_output_time.pkl', 'rb') as file:
            PWLs_loaded, margin_loaded, Box_loaded = pickle.load(file)

        PWLs = PWLs_loaded
        margin_loaded = margin_loaded



        running_time_3, margin_result = optimize_boundingbox(PWLs, swarm, margin_loaded, Box_loaded, test_list, num_segs)
        margins.append(margin_result)
        print(f"The running time is {running_time_3}.")
        total_running_time = running_time_1 + running_time_2 + running_time_3
        total_time.append(total_running_time)
        print(f"Round {num_round} is finished.")
        if margin_result >= 0:
            print("The solution is feasible.")
            break



    final_time = sum(total_time)
    print(total_time)
    print(final_time)
    print(margins)

    _, plots = task_environment()
    plt.rcParams["figure.figsize"] = [6.4, 6.4]
    plt.rcParams['axes.titlesize'] = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            pf.plot_polygon_ge(A, b, color=plot[1])


    plt.gca().set_aspect('equal', adjustable='box')

    with open('PWLs_output_time.pkl', 'rb') as file:
        PWLs, margin_loaded, Box_loaded = pickle.load(file)
    if len(PWLs) <= 4:
        colors = ['k', np.array([153, 0, 71]) / 255, np.array([6, 0, 153]) / 255, np.array([0, 150, 0]) / 255]
    else:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 0.85, len(PWLs))]

    for i in range(len(PWLs)):
        PWL = PWLs[i]
        ax.plot([P[0][0] for P in PWL], [P[0][1] for P in PWL], '-', color=colors[i])
        ax.plot(PWL[-1][0][0], PWL[-1][0][1], '*', color=colors[i])
        ax.plot(PWL[0][0][0], PWL[0][0][1], 'o', color=colors[i])

    
    plt.show()


