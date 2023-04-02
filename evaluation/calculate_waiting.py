import sumolib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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

    # print('Count: Total-{}, Default-{}, Ambulance-{}, FuelTruck-{}, Trailer-{}'.format(count_vehicle,
    #                                                                                    count_vehicle_default,
    #                                                                                    count_Ambulance, count_trailer,
    #                                                                                    count_fueltruck))
    return sum_waiting_time / count_vehicle, sum_waiting_time_default / count_vehicle_default, \
           sum_waiting_time_ambulance / max(1, count_Ambulance), sum_waiting_time_fueltruck / max(1, count_fueltruck), \
           sum_waiting_time_trailer / max(count_trailer, 1), count_vehicle, count_vehicle_default,\
           count_Ambulance, count_trailer, count_fueltruck


def draw_total(total, iteration):
    df = pd.DataFrame({'Training iteration': [i for i in range(0, iteration)], 'Average waiting time': total})
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
    output_name = "experiments/Qlearning/ImplicitObservation/ImplicitObservation-QL-tripinfo-proposed-run2-iter"
    # output_name = "experiments/Qlearning/ImportantObservation/ImportantObservation-QL-tripinfo-proposed-run6-iter"
    # output_name = "experiments/Qlearning/SampleObservation/SampleObservation-QL-tripinfo-proposed-run1-iter"
    output_name = "experiments/DQN/ImplicitObservation/ImplicitObservation-DQN-tripinfo-proposed-run1-iter"
    iteration = 185

    total = []
    default = []
    ambulance = []
    fueltruck = []
    trailer = []

    count_total = []
    count_default = []
    count_ambulance = []
    count_fueltruck = []
    count_trailer = []

    for i in range(0, iteration):
        avg_waiting_total, avg_waiting_default, avg_waiting_ambulance, avg_waiting_fueltruck, avg_waiting_trailer, \
        count_vehicle, count_vehicle_default,count_vehicle_Ambulance, count_vehicle_trailer, count_vehicle_fueltruck \
            = get_waiting(output_name+str(i)+'.xml')
        total.append(avg_waiting_total)
        default.append(avg_waiting_default)
        ambulance.append(avg_waiting_ambulance)
        fueltruck.append(avg_waiting_fueltruck)
        trailer.append(avg_waiting_trailer)

        count_total.append(count_vehicle)
        count_default.append(count_vehicle_default)
        count_ambulance.append(count_vehicle_Ambulance)
        count_fueltruck.append(count_vehicle_fueltruck)
        count_trailer.append(count_vehicle_trailer)

    print('Count vehicle, average:')
    print('Total: {}, Default: {}, Ambulance: {}, FuelTruck: {}, Trailer: {}'.format(np.mean(count_total),
                                                                                     np.mean(count_default),
                                                                                     np.mean(count_ambulance),
                                                                                     np.mean(count_fueltruck),
                                                                                     np.mean(count_trailer)))

    print('Average waiting time:')
    df = pd.DataFrame({'Total': total, 'Default': default, 'Ambulance': ambulance, 'FuelTruck': fueltruck, 'Trailer': trailer})
    print(df)
    print('Total:', np.mean(total), 'Default: ', np.mean(default), 'Ambulance: ', np.mean(ambulance),
          'FuelTrack: ', np.mean(fueltruck), 'Trailer: ', np.mean(trailer))

    print('Average waiting time (last 50):')
    print('Total:', np.mean(total[:-50]), 'Default: ', np.mean(default[:-50]), 'Ambulance: ', np.mean(ambulance[:-50]),
          'FuelTrack: ', np.mean(fueltruck[:-50]), 'Trailer: ', np.mean(trailer[:-50]))

    draw_total(total, iteration)
