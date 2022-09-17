import numpy as np

def generate_our_IP_demo():
    Adict = []
    Bdict = []
    # num of edges of trip to vehicle
    num_e = 10
    # num of requests
    num_x = 4
    # num of vehicles
    num_v = 2
    # num of requests
    num_r = 4
    # num of trips
    num_t = 7
    # total num of vairables
    num_var = num_e + num_x

    # random num of trips can be assigned to vehicles
    num_v1_t = 3 #np.random.randint(3, 7, size=1)[0]
    num_v2_t = num_e - num_v1_t
    # print(num_v1_t,num_v2_t)

    # constraint 1: each vehicle has at most 1 trip
    a1 = np.zeros(num_var)
    a1[0:num_v1_t] = 1
    # print(a1)
    b1 = np.ones(1)[0]
    Adict.append(a1)
    Bdict.append(b1)

    a2 = np.zeros(num_var)
    a2[num_v1_t:10] = 1
    # print(a2)
    b2 = np.ones(1)[0]
    Adict.append(a2)
    Bdict.append(b2)

    # print(Adict)
    # print(Bdict)

    # constraint 2: each request is assigned to a vehicle or ignored
    # for request 1
    idx_list = [[3, 7, 10], [0, 4, 9, 11], [1, 5, 2, 8, 12], [6, 7, 9, 13]]
    for i in range(num_r):
        a = np.zeros(num_var)
        for j in range(len(idx_list[i])):
            a[idx_list[i][j]] = 1
        # print(a)
        b = np.ones(1)[0]
        Adict.append(a)
        Bdict.append(b)
        Adict.append(-a)
        Bdict.append(-b)

    # print(Adict)
    # print(Bdict)

    # print(len(Adict))

    # binary constraints
    for i in range(2 * num_var):
        if i < num_var:
            a = np.zeros(num_var)
            a[i] = 1
            b = np.ones(1)[0]
        else:
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
    c = np.random.rand(1, 14)
    c[0][num_e:] = ck_0
    A = np.array(Adict)
    B = np.array(Bdict)

    return A,B, c

generate_our_IP_demo()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-c', type=int, default=38)
    parser.add_argument('--num-v', type=int, default=14)
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
    # print(b)
    print("size of c : ", c.shape)
    print(c)

    np.save(logdir + '/A', A)
    np.save(logdir + '/b', b)
    np.save(logdir + '/c', c)

    print('test instance generated!')