import numpy as np
def generate_our_IP_demo(num_v=2, num_r=4, num_trip=7):
    Adict = []
    Bdict = []
    # num of edges of trip to vehicle
    # num_e = 10
    # num of vehicles
    # num_v = 2
    # num of requests
    # num_r = 4
    # num of trips
    num_t = 7
    # total num of vairables
    # num_var = num_e + num_x
    num_e = num_v*num_trip
    num_var = num_v*num_trip+num_r

    # random num of trips can be assigned to vehicles
    num_v1_t = 3 #np.random.randint(3, 7, size=1)[0]
    t = [2,3,6]#np.random.randint(0, 7, size=num_v1_t)

    # constraint 1: each vehicle has at most 1 trip
    a1 = np.zeros(num_var)
    for i in range(num_v1_t):
        a1[t[i]] = 1
    # a1[0:num_v1_t] = 1
    b1 = np.ones(1)[0]
    Adict.append(a1)
    Bdict.append(b1)

    num_v2_t = 7
    a2 = np.zeros(num_var)
    a2[num_v2_t:14] = 1
    # print(a2)
    b2 = np.ones(1)[0]
    Adict.append(a2)
    Bdict.append(b2)

    # print(Adict)
    # print(Bdict)

    # constraint 2: each request is assigned to a vehicle or ignored
    # for request 1
    idx_list = [[7, 11,14], [6, 8, 13,15], [2,9, 12,16], [3,6, 10, 11,13,17]]
    for i in range(num_r):
        a = np.zeros(num_var)
        for j in range(len(idx_list[i])):
            a[idx_list[i][j]] = 1

        b = np.ones(1)[0]
        Adict.append(a)
        Bdict.append(b)
        # Adict.append(-a)
        # Bdict.append(-b)

    # print(Adict)
    # print(Bdict)

    # print(len(Adict))

    # binary constraints
    for i in range(num_var):

        a = np.zeros(num_var)
        a[i] = 1
        b = np.ones(1)[0]
        Adict.append(a)
        Bdict.append(b)

        a = np.zeros(num_var)
        a[i - num_var] = -1
        b = np.zeros(1)[0]
        Adict.append(a)
        Bdict.append(b)
    # print(Adict)
    # print(Bdict)

    # cost function
    # penalization for ignored request
    ck_0 = 5
    # random edge cost for trip to vehicle
    # c = np.random.randint(1, 10, size=num_var)
    c = np.random.rand(1, num_var)
    c[num_e:18] = 5
    A = np.array(Adict)
    B = np.array(Bdict)

    return A,B, c

# generate_our_IP_demo()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-c', type=int, default=46)
    parser.add_argument('--num-v', type=int, default=18)
    parser.add_argument('--num-instances', type=int, default=100)
    args = parser.parse_args()

    n = args.num_v
    m = args.num_c
    num_instances = args.num_instances

    import os

    logdir = 'instances/our_test_IP/randomip_n{}_m{}'.format(n, m)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print('start generating instances...')

    A,b,c=generate_our_IP_demo()
    print("size of A : ", A.shape)
    print("size of B : ", b.shape)
    print(b)
    print(c)
    print("size of c : ", c.shape)

    np.save(logdir + '/A', A)
    np.save(logdir + '/b', b)
    np.save(logdir + '/c', c)

    print('test instance generated!')