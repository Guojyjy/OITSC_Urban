import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    output_name = "experiments/Qlearning/ImplicitObservation/ImplicitObservation-QL-output-baseline_run"
    iteration = 4

    reward_iter = []
    for i in range(1, iteration):
        train_file = output_name+str(i)+'.csv'
        reward = sum_reward(train_file)
        print(i, reward, train_file)
        reward_iter.append(reward)

    draw_reward(reward_iter, iteration)
