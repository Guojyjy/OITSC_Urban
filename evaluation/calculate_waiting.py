import sumolib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_waiting(tripinfo_file):
    count_vehicle = 0
    sum_waiting_time = 0

    count_vehicle_default = 0
    sum_waiting_time_default = 0
    count_trailer = 0
    sum_waiting_time_trailer = 0
    count_fueltruck = 0
    sum_waiting_time_fueltruck = 0
    count_Ambulance = 0
    sum_waiting_time_ambulance = 0

    for trip in sumolib.output.parse(tripinfo_file, ['tripinfo']):
        count_vehicle += 1
        sum_waiting_time += float(trip.waitingTime)
        if trip.vType == 'DEFAULT_VEHTYPE':
            count_vehicle_default += 1
            sum_waiting_time_default += float(trip.waitingTime)
        elif trip.vType == 'trailer':
            count_trailer += 1
            sum_waiting_time_trailer += float(trip.waitingTime)
        elif trip.vType == 'fueltruck':
            count_fueltruck += 1
            sum_waiting_time_fueltruck += float(trip.waitingTime)
        else:
            count_Ambulance += 1
            sum_waiting_time_ambulance += float(trip.waitingTime)

    return sum_waiting_time / count_vehicle, sum_waiting_time_default / count_vehicle_default, \
           sum_waiting_time_ambulance / count_Ambulance, sum_waiting_time_fueltruck / count_fueltruck, \
           sum_waiting_time_trailer / count_trailer


def draw_total(total, iteration):
    df = pd.DataFrame({'Training iteration': [i for i in range(1, iteration)], 'Average waiting time': total})
    fig, ax = plt.subplots()
    sns.lineplot(x=df['Training iteration'], y=df['Average waiting time'], ax=ax)
    sns.color_palette('bright')
    # plt.legend(title='', loc='lower right', fontsize='13')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.xlim(-5, 150)
    ax.set_xlabel("Training Iteration", fontsize=14)
    ax.set_ylabel("Average waiting time (total)", fontsize=14)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    output_name = "experiments/Qlearning/ImplicitObservation/ImplicitObservation-QL-tripinfo-baseline-run"
    iteration = 2

    total = []
    default = []
    ambulance = []
    fueltrack = []
    trailer = []
    for i in range(1, iteration):
        avg_waiting_total, avg_waiting_default, avg_waiting_ambulance, avg_waiting_fueltruck, avg_waiting_trailer \
            = get_waiting(output_name+str(i)+'.xml')
        total.append(avg_waiting_total)
        default.append(avg_waiting_default)
        ambulance.append(avg_waiting_ambulance)
        fueltrack.append(avg_waiting_fueltruck)
        trailer.append(avg_waiting_trailer)

    print('Average waiting time:')
    df = pd.DataFrame({'Total': total, 'Default': default, 'Ambulance': ambulance, 'FuelTrack': fueltrack, 'Trailer': trailer})
    print(df)

    draw_total(total, iteration)
