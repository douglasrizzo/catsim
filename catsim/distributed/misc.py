import os

_ventilator_port = '5557'
_sink_port = '5558'
_data_port = '5559'
default_ip = '10.24.2.18'


def ping(ip):
    response = os.system('ping -c 1 ' + str(ip))
    return response == 0