from solverutils import computeoptimaltab, generatecutzeroth
from gymenv_v2 import *
from rl_cuts import *
import cvxpy as cp
# from scipyutils import ScipyLinProgSolve
import matplotlib as plt


# variable
n = 8
# constrain
m = 6

PATH = "models/Qiming/our_config_best_model_2_6.pt"
actor = torch.load(PATH)
done = False
training = False
explore = False

# gym loop
# To keep a record of states actions and reward for each episode
obss_constraint = []  # states
obss_cuts = []
acts = []
acts_max = []
rews = []
states = []
repisode = 0
iters = 0
reward_type = 'obj'
timelimit = 50
# np.random.seed(1)
# random.seed(1)

# s may be selected artificially: it is the origin form of IP from VRP
load_dir = "instances/randomip_n2_m10"
idx = 99
print('loading training instances, dir {} idx {}'.format(load_dir, idx))
A = np.load('{}/A_{}.npy'.format(load_dir, idx))
b = np.load('{}/b_{}.npy'.format(load_dir, idx))
c = np.load('{}/c_{}.npy'.format(load_dir, idx))


# test instance
load_dir_test = 'instances/our_test_IP/randomip_n14_m38'
# A = np.load('{}/A.npy'.format(load_dir_test))
# b = np.load('{}/b.npy'.format(load_dir_test))
# c = np.load('{}/c.npy'.format(load_dir_test))
# test_env = timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type=reward_type), timelimit)
# s = test_env.reset()

# simple test
A_t =[[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1],
         [1, 2, 2, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 2, 1]]
A_t = [[4, 5], [8, 5], [1, 0], [0, 1], [-1, 0], [0, -1]]
b_t = [1, 1, 1, 1, 5, 5]
b_t = [16, 20, 2, 3, 0, 0]
# for i in range(8):
#     a = np.zeros(8)
#     a[i] = 1
#     b = np.ones(1)[0]
#     A_t.append(a)
#     b_t.append(b)
#
#     a = np.zeros(8)
#     a[i - 8] = -1
#     b = np.zeros(1)[0]
#     A_t.append(a)
#     b_t.append(b)
A_t = np.array(A_t)
b_t = np.array(b_t).T
c_t = abs(np.random.random(8))
# c_t = np.array([0.01,0.02,0.03,0.04,0.09,0.10,0.12,0.15])
# c_t = np.array([-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4])
c_t = np.array([1,1,1,1,4,4,4,4])
# c_t = np.array([1,2,1,2,1,2,1,2])
c_t = np.array([6, 5])

# set environment
# test_env = timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type=reward_type), timelimit)
test_env = timelimit_wrapper(GurobiOriginalEnv(A_t, b_t, c_t, solution=None, reward_type=reward_type), timelimit)

s = test_env.reset()

# rollout
while not done:
    iters += 1
    A, b, c0, cuts_a, cuts_b = s
    available_cuts_t = np.concatenate((cuts_a, cuts_b[:, None]), axis=1)
    A, b, cuts_a, cuts_b = normalization(A, b, cuts_a, cuts_b)

    # concatenate [a, b] [e, d]
    curr_constraints = np.concatenate((A, b[:, None]), axis=1)
    available_cuts = np.concatenate((cuts_a, cuts_b[:, None]), axis=1)

    # compute probability distribution using the policy network
    prob = actor.compute_prob(curr_constraints, available_cuts)
    # print(prob)
    prob /= np.sum(prob)

    # explore_rate = min_explore_rate + \
    #                (max_explore_rate - min_explore_rate) * np.exp(-explore_decay_rate * (e))

    # epsilon greedy for exploration
    if training and explore:
        random_num = random.uniform(0, 1)
        if random_num <= explore_rate:
            a = np.random.randint(0, s[-1].size, 1)
        else:
            # a = np.argmax(prob)
            a = [np.random.choice(s[-1].size, p=prob.flatten())]
    else:
        # for testing case, only sample action
        a = [np.random.choice(s[-1].size, p=prob.flatten())]


    a_m = np.argmax(prob)
    a_max = [np.argmax(prob)]
    acts.append(available_cuts_t[a_m])
    # print(a)
    print(a_max)

    new_state, r, done, _ = test_env.step(list(a_max))
    obss_constraint.append(curr_constraints)
    obss_cuts.append(available_cuts)
    # acts.append(available_cuts[acts_max])
    acts_max.append(a_max)
    rews.append(r)
    s = new_state
    states.append(new_state)
    repisode += r

def INtsolve(A,b,c):
	x = cp.Variable(2, integer=True)
	# c = abs(np.random.random(8))
	prob = cp.Problem(cp.Maximize(c * x),
					  [A @ x <= b, x >= 0])
	prob.solve()
	print("\nThe optimal value is", prob.value)
	print("A solution x is")
	print(x.value)

def compute_solution():
    s0 = test_env.reset()
    A, b, c, cuts_a, cuts_b = s0
    print("size of A : ", A.shape)
    print("size of B : ", b.shape)
    print("size of c : ", c.shape)
    # print(A,b)
    m, n = A.shape
    assert m == b.size and n == c.size
    # INtsolve(A, b, c)
    A_tilde = np.column_stack((A, np.eye(m)))
    b_tilde = b
    c_tilde = np.append(c, np.zeros(m))
    A, b, c = A_tilde, b_tilde, c_tilde
    print("size of A for solution : ", A.shape)
    print("size of B for solution: ", b.shape)
    print("size of c for solution: ", c.shape)
    # compute gaps
    objint_o, solution_int = gurobiutils.GurobiIntSolve(A, b, c)


    objlp_o, solution_lp, _, _ = gurobiutils.GurobiSolve(A, b, c)

    print("oringin IP obj = "+str(objint_o))
    print("oringin IP solution = " + str(solution_int))
    print("oringin LP obj = " + str(objlp_o))
    print("oringin LP solution = " + str(solution_lp))
    # print("oringin LP obj = "+str(objlp_o))

    s = states[-1]
    A, b, c, cuts_a, cuts_b = s
    print("size of A : ", A.shape)
    # print(A,b)
    print("size of B : ", b.shape)
    print("size of c : ", c.shape)
    # INtsolve(A,b,c)
    m, n = A.shape
    assert m == b.size and n == c.size
    A_tilde = np.column_stack((A, np.eye(m)))
    b_tilde = b
    c_tilde = np.append(c, np.zeros(m))
    A, b, c = A_tilde, b_tilde, c_tilde
    # compute gaps
    print("size of A for solution : ", A.shape)
    print("size of B for solution: ", b.shape)
    print("size of c for solution: ", c.shape)
    objint_f, solution_int = gurobiutils.GurobiIntSolve(A, b, c)

    objlp_f, solution_lp, _, _ = gurobiutils.GurobiSolve(A, b, c)

    print("final IP obj = " + str(objint_f))
    print("final IP solution = " + str(solution_int))
    print("final LP obj = " + str(objlp_f))
    print("final LP solution = " + str(solution_lp))
    # print("final LP obj = " + str(objlp_f))
# a = acts_max

compute_solution()

print(acts)
def compute_gap(iter):
    if iters == 0:
        s0 = test_env.reset()
        A, b, c, cuts_a, cuts_b = s0
        m, n = A.shape
        assert m == b.size and n == c.size
        A_tilde = np.column_stack((A, np.eye(m)))
        b_tilde = b
        c_tilde = np.append(c, np.zeros(m))
        A, b, c = A_tilde, b_tilde, c_tilde
        # compute gaps
        objint, solution_int = gurobiutils.GurobiIntSolve(A, b, c)
        objlp, solution_lp, _, _ = gurobiutils.GurobiSolve(A, b, c)
        Gap = objint - objlp
    else:
        s = states[iter]
        A, b, c, cuts_a, cuts_b = s
        m, n = A.shape
        assert m == b.size and n == c.size
        A_tilde = np.column_stack((A, np.eye(m)))
        b_tilde = b
        c_tilde = np.append(c, np.zeros(m))
        A, b, c = A_tilde, b_tilde, c_tilde
        # compute gaps
        objint, solution_int = gurobiutils.GurobiIntSolve(A, b, c)
        objlp, solution_lp, _, _ = gurobiutils.GurobiSolve(A, b, c)
        Gap = objint - objlp
    return Gap

def IGC(current_iter):
    Gap_0 = compute_gap(0)
    Gap_t = compute_gap(current_iter)
    return (Gap_0-Gap_t)/Gap_0

def compute_sol_in_gorubi():
    s0 = test_env.reset()
    A, b, c, cuts_a, cuts_b = s0
    print("size of A : ", A.shape)
    print("size of B : ", b.shape)
    print("size of c : ", c.shape)
    # print(A,b)
    m, n = A.shape
    assert m == b.size and n == c.size
    # INtsolve(A, b, c)
    A_tilde = np.column_stack((A, np.eye(m)))
    b_tilde = b
    c_tilde = np.append(c, np.zeros(m))
    A, b, c = A_tilde, b_tilde, c_tilde
    print("size of A for solution : ", A.shape)
    print("size of B for solution: ", b.shape)
    print("size of c for solution: ", c.shape)
    # compute gaps
    objint_o, solution_int = gurobiutils.GurobiIntSolve(A, b, c)


    objlp_o, solution_lp, _, _ = gurobiutils.GurobiSolve(A, b, c)

    print("oringin IP obj = "+str(objint_o))
    print("oringin IP solution = " + str(solution_int))
    print("oringin LP obj = " + str(objlp_o))
    print("oringin LP solution = " + str(solution_lp))
    # print("oringin LP obj = "+str(objlp_o))
# compute_sol_in_gorubi()





