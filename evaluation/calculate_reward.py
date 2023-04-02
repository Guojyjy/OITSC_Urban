import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def sum_reward(train_file):
    df = pd.read_csv(train_file)
    return sum(df['avg_reward'])


def draw_reward(reward_iter, iteration):
    df = pd.DataFrame({'Training iteration': [i for i in range(1, iteration)], 'Reward': reward_iter})
    fig, ax = plt.subplots()
    sns.lineplot(x=df['Training iteration'], y=df['Reward'], ax=ax)
    sns.color_palette('bright')
    # plt.legend(title='', loc='lower right', fontsize='13')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.xlim(-5, 150)
    ax.set_xlabel("Training Iteration", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    output_name = "experiments/Qlearning/ImplicitObservation/ImplicitObservation-QL-output-proposed-run1_iter"
    # output_name = "experiments/Qlearning/ImportantObservation/ImportantObservation-QL-output-baseline-run3_iter"

    iteration = 999

    reward_iter = []
    for i in range(1, iteration):
        train_file = output_name+str(i)+'.csv'
        reward = sum_reward(train_file)
        # print(i, reward, train_file)
        reward_iter.append(reward)

    print("Reward average: {}".format(np.mean(reward_iter)))
    draw_reward(reward_iter, iteration)
