# -*- coding: utf-8 -*-

import sys
import zmq
import time
import json
import socket
import pandas
import smtplib
import platform
import numpy as np
import catsim.misc.results
from multiprocessing import Process

from catsim.distributed.misc import ventilator_port, sink_port, data_port, default_ip, ping


def relevantResults():
    # load existing CAT results and all clustering results
    clusterResultsAux = catsim.misc.results.loadClusterResults('results_masters.csv')

    # filter clustering results to those with n. of groups multiples of ten
    clusterResults = pandas.DataFrame(data=None, columns=clusterResultsAux.columns)

    for i in range(10, 251, 10):
        # if i == 250:
        #     i = 249
        # hello world
        aux = clusterResultsAux[(clusterResultsAux['Nº grupos'] == i) & (clusterResultsAux['Algoritmo'] == 'WK-means')]
        clusterResults = clusterResults.append(aux.ix[aux['Variância'].idxmin()])
        clusterResults = clusterResults.append(aux.ix[aux['Dunn'].idxmax()])

    return clusterResults


def cases():
    # Itens Método Índice r.max
    cases = pandas.DataFrame(columns=['Qtd. itens', 'Método', 'Índice', 'r. max', 'Classificações'])
    cases[['Qtd. itens', 'Índice']] = cases[['Qtd. itens', 'Índice']].astype(np.int64)
    cases['r. max'] = cases['r. max'].astype(np.float64)
    clusterResults = relevantResults()
    catResults = catsim.misc.results.loadGlobalCATResults('cat_results.csv').drop_duplicates(subset=['Índice', 'r. max']).sort(['Índice', 'r. max'])
    canonical_rmaxes = np.round(np.linspace(0.1, 1, 10, dtype=float), 1)
    itens = [20]
    methods = ['item_info']

    processed_cases = catResults[['Qtd. itens', 'Método', 'Índice', 'r. max']]

    for method in methods:
        for item in itens:
            for rmax in canonical_rmaxes:
                for index, clusterResult in clusterResults.iterrows():
                    if processed_cases[
                        (processed_cases['Qtd. itens'] == item) & (processed_cases['Método'] == method) & (processed_cases['Índice'] == index) & (
                            processed_cases['r. max'] == rmax)
                    ].shape[0] == 0:
                        cluster = str(clusterResult['Classificações']).strip('[]').replace(',', '')
                        cases = cases.append(
                            pandas.DataFrame(
                                {
                                    'Qtd. itens': item,
                                    'Método': method,
                                    'Índice': index,
                                    'r. max': rmax,
                                    'Classificações': cluster
                                },
                                index=[0]
                            )
                        )

    return cases


def send_email():
    fromaddr = 'douglasrizzom@gmail.com'
    toaddrs = 'douglasrizzom@gmail.com'
    msg = "\r\n".join(["From: douglasrizzom@gmail.com", "To: douglasrizzom@gmail.com", "Subject: Finish", "", "The simulations finished!"])
    username = 'douglasrizzom@gmail.com'
    password = 'douglasobeso'

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(username, password)
        server.sendmail(fromaddr, toaddrs, msg)
        server.quit()
        print('E-mail sent')
    except:
        print('Failed to send e-mail')


def indexes_and_rmaxes(clusterResults):
    catResults = catsim.misc.results.loadGlobalCATResults('cat_results.csv').drop_duplicates(subset=['Índice', 'r. max']).sort(['Índice', 'r. max'])

    canonical_rmaxes = np.round(np.linspace(0.1, 1, 10, dtype=float), 1)

    indexes = []
    rmaxes = []

    # iterate through all cluster results that shoulb be simulated
    for index in clusterResults.index:
        # get rmaxes already simulated for this index
        simulated_rmaxes = list(np.round(catResults[catResults['Índice'] == index]['r. max'], 1))

        # filter only those rmax values that were not simulate
        not_simulated_rmaxes = []
        for i in canonical_rmaxes:
            if np.round(i, 1) not in np.round(simulated_rmaxes, 1):
                not_simulated_rmaxes.append(np.round(i, 1))
                if index not in indexes:
                    indexes.append(index)

        rmaxes.append(list(np.round(not_simulated_rmaxes, 1)))

    return indexes, rmaxes


def ventilate_datasets(datasets):
    if type(datasets) != dict:
        raise ValueError('datasets must be a dictionary containing name and data')
    context = zmq.Context()

    # Socket to send messages on
    sender = context.socket(zmq.REP)
    sender.bind("tcp://*:" + data_port)
    print("Sending datasets to workers...")

    while True:
        workerName = str(sender.recv())
        print('....Sending data to ' + workerName + ' -- ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        sender.send_pyobj(datasets)

    # Give 0MQ time to deliver
    time.sleep(1)


def ventilate_cases():
    context = zmq.Context()

    workload = cases()

    # socket to send the server IP to the discoverer
    ipSender = context.socket(zmq.REQ)

    if len(sys.argv) == 0:
        if platform.system() == 'Linux':
            print('Trying to ping default server at ' + default_ip + '...')
            if ping(default_ip):
                ipSender.connect('tcp://' + default_ip + ':5554')
            else:
                print('...Ping failed, connecting to localhost')
                ipSender.connect('tcp://localhost:5554')
        else:
            ipSender.connect('tcp://' + default_ip + ':5554')

    else:
        ipSender.connect('tcp://localhost:5554')

    print('Sending address to discoverer...')

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("gmail.com", 80))
    ipSender.send_string(s.getsockname()[0])
    s.close()
    ipSender.recv()

    # Socket to send messages on
    sender = context.socket(zmq.REP)
    sender.bind("tcp://*:" + ventilator_port)

    # Socket with direct access to the sink: used to synchronize start of batch
    sink = context.socket(zmq.PUSH)
    sink.connect("tcp://localhost:" + sink_port)
    print("Sending tasks to workers...")

    # ventilate cases forever, until all results are received
    while True:
        if workload.shape[0] == 0:
            break

        for work_index, work in workload.iterrows():
            workerName = str(sender.recv())
            # buf = StringIO()
            # # special treatment for the cluster results column
            # work.to_csv(buf)
            # buf.seek(0)

            buf = json.dumps(work.to_dict())

            print('....Sending case ' + str(work_index) + ' to ' + workerName + ' -- ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            sender.send_pyobj(buf)

        workload = cases()

    send_email()


def start_sink():
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:" + sink_port)

    # Wait for start of batch
    # s = receiver.recv()
    # while s != b'0':
    #     s = receiver.recv()

    # Start our clock now
    tstart = time.time()

    print('Receiving results from workers...')

    while True:
        s = json.loads(receiver.recv())
        globalResults = s['globalResults']
        # localResults = s['localResults']

        print(
            '....Received results for case ' + format(s['index']) + ', r. max ' + format(globalResults['r_max']) + ' -- ' + time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(time.time())
            )
        )

        catsim.misc.results.saveGlobalCATResults(
            s['index'], s['date'], s['method'], s['time'], globalResults['Nº de grupos'], globalResults['Qtd. Itens'], globalResults['RMSE'],
            globalResults['Overlap'], globalResults['r_max'], 'cat_results.csv'
        )

        # for result in localResults:
        #     catsim.misc.results.saveLocalCATResults(
        #         s['index'],
        #         result['Theta'],
        #         result['Est. Theta'],
        #         result['Id. Itens'],
        #         result['r_max'],
        #         'individual_cat_results.csv')

        # Calculate and report duration of batch
    tend = time.time()
    print("Total elapsed time: %d msec" % ((tend - tstart) * 1000))


if __name__ == '__main__':
    Process(target=ventilate_datasets).start()
    Process(target=start_sink).start()
    Process(target=ventilate_cases).start()
