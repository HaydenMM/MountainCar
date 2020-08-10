# Hayden Moore
# Intro to Machine Learning - HW8
# Q Learning

from environment import MountainCar
import sys
import numpy


def greedy(s, w, epsilon, car, b):
    space = car.action_space
    new_a = numpy.ones(space, dtype=float)
    new_a = new_a * epsilon/space

    next_aa = []
    for action in range(car.action_space):
        next_aa.append(q(s,action, w, b))
    next_a = numpy.argmax(next_aa)

    prob = 1 - epsilon
    new_a[next_a] = new_a[next_a] + prob
    return numpy.random.choice(car.action_space, p=new_a)


def dict_to_np(dict):
    nump = [0, 0]
    nump[0] = dict[0]
    nump[1] = dict[1]
    nump = numpy.array(nump)

    return nump


def q(s, a, w, b):
    val = numpy.dot(s, w[a]) + b
    return val


def main():
    # get command line arguments
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    b = 0

    car = MountainCar(mode)
    w = numpy.zeros((car.action_space, car.state_space))
    returns = numpy.zeros(episodes)
    s = car.reset()
    s = dict_to_np(s)

    for e in range(episodes):

        done = False
        for it in range(max_iterations):
            if it == max_iterations or done:
                s = car.reset()
                s = dict_to_np(s)
                break

            a = greedy(s, w, epsilon, car, b)
            next_s, r, done = car.step(a)
            next_s = dict_to_np(next_s)
            next_action = greedy(next_s, w, epsilon, car, b)
            returns[e] = returns[e] + r
            q_learn = r + gamma * q(next_s, next_action, w, b)
            error = q(s, a, w, b) - q_learn
            gradient = numpy.dot(error, s)
            w[a] = w[a] - (learning_rate * gradient)
            b = b - learning_rate * error

            s = next_s

    with open(weight_out, 'w') as f:
        w = w.transpose()
        f.write(str(b) + '\n')
        for ww in w:
            for i in ww:
                f.write(str(i) + '\n')

    with open(returns_out, 'w') as f:
        for r in returns:
            f.write(str(r) + '\n')


if __name__ == "__main__":
    main()
