# -*- coding: utf-8 -*-

import zmq
import sys
import json
import time
import numpy as np
import catsim.cat.simulate
import multiprocessing as mp
from socket import gethostname
from multiprocessing import Process
from catsim.distributed import defaultIP as serverIp


def start():
    hostname = gethostname()
    context = zmq.Context()

    # Socket to receive messages on
    cluster_socket = context.socket(zmq.REQ)
    cluster_socket.connect('tcp://' + serverIp + ':5557')

    # Socket to send messages to
    sink_socket = context.socket(zmq.PUSH)
    sink_socket.connect('tcp://' + serverIp + ':5558')

    # Socket to get the datasets from
    data_socket = context.socket(zmq.REQ)
    data_socket.connect('tcp://' + serverIp + ':5559')

    print('....Waiting for datasets....')
    data_socket.send_string(gethostname())
    # só vai fazer experimentos com a base sintética
    x = json.loads(data_socket.recv())[0][1]
    print('........Done!\n....Waiting for tasks....')

    # Process tasks forever
    while True:
        cluster_socket.send_string(hostname)
        # string_row = json.loads(cluster_socket.recv())
        # s = pandas.Series.from_csv(string_row, parse_dates=False)
        s = json.loads(cluster_socket.recv())
        s[['Qtd. itens', 'Índice']] = s[['Qtd. itens', 'Índice']].astype(np.int64)
        s['r. max'] = np.float64(s['r. max'])
        s['Classificações'] = ' '.join(s['Classificações'].split()).split(' ')
        s['Classificações'] = np.array(s['Classificações'])

        print('....Processing case ' + str(s))
        # tries ten times to simulate a test application. this is necessary
        # due to convergence problems, which were mostly solved in the
        # past, but still, just a precaution so that automation of the
        # tests is guaranteed
        for i in range(10):
            try:
                t1 = time.time()
                globalResults, localResults = catsim.cat.simulate.simCAT(
                    items=x,
                    clusters=s['Classificações'],
                    examinees=(x.shape[0] * 10),
                    n_itens=s['Qtd. itens'],
                    r_max=s['r. max'],
                    method=s['Método'],
                    optimization='DE',
                    r_control='passive'
                )
                t2 = time.time() - t1

                sinkMessage = {
                    'index': s['Índice'],
                    'date': t1,
                    'time': t2,
                    'method': s['Método'],
                    'dataset': 'Sintética',
                    'globalResults': globalResults,
                    'localResults': localResults
                }

                sink_socket.send_pyobj(sinkMessage)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                raise


if __name__ == '__main__':
    numWorkers = 0
    if len(sys.argv) > 1:
        numWorkers = mp.cpu_count() if int(sys.argv[1]) > mp.cpu_count() else int(sys.argv[1])
    else:
        numWorkers = mp.cpu_count()

    for cpu in range(numWorkers):
        Process(target=start, name=str(cpu)).start()
