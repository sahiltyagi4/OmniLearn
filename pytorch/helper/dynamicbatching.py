
import os
import time
import logging
import subprocess

class DynamicHeterogeneityEmulator(object):

    # model_name = resnet18/resnet50/vgg11/alexnet/gpt2
    # sync_mode = ASP/BSP
    def __init__(self, model_name, sync_mode, cpulog_file, id):
        self.model_name = model_name
        self.sync_mode = sync_mode
        self.cpu_log = cpulog_file

        if self.sync_mode == 'BSP':
            self.worker_name = 'worker' + str(id)
        elif self.sync_mode == 'ASP':
            self.worker_name = 'aspworker' + str(id)

        self.hl_routine = self.getDynamicHLRoutine()

    def getContainerCPUConfigurations(self, h_level):
        container_cpuconf = {}
        if self.sync_mode == 'ASP':

            if h_level == 'HL1':
                container_cpuconf["aspworker1"] = "0-7"
                container_cpuconf["aspworker2"] = "8-17"
                container_cpuconf["aspworker3"] = "18-27"
                container_cpuconf["aspworker4"] = "28-37"
                container_cpuconf["aspworker5"] = "38-47"

            elif h_level == 'HL2':
                container_cpuconf["aspworker1"] = "0-7"
                container_cpuconf["aspworker2"] = "8-15"
                container_cpuconf["aspworker3"] = "16-23"
                container_cpuconf["aspworker4"] = "24-31"
                container_cpuconf["aspworker5"] = "32-47"

            elif h_level == 'HL4':
                container_cpuconf["aspworker1"] = "0-7"
                container_cpuconf["aspworker2"] = "8-17"
                container_cpuconf["aspworker3"] = "18-27"
                container_cpuconf["aspworker4"] = "28-31"
                container_cpuconf["aspworker5"] = "32-47"

            elif h_level == 'HL8':
                container_cpuconf["aspworker1"] = "0-7"
                container_cpuconf["aspworker2"] = "8-11"
                container_cpuconf["aspworker3"] = "12-15"
                container_cpuconf["aspworker4"] = "16-19"
                container_cpuconf["aspworker5"] = "20-47"

        elif self.sync_mode == 'BSP':

            if h_level == 'HL1':
                container_cpuconf["worker1"] = "0-11"
                container_cpuconf["worker2"] = "12-23"
                container_cpuconf["worker3"] = "24-35"
                container_cpuconf["worker4"] = "36-47"

            elif h_level == 'HL2':
                container_cpuconf["worker1"] = "0-11"
                container_cpuconf["worker2"] = "12-23"
                container_cpuconf["worker3"] = "24-31"
                container_cpuconf["worker4"] = "32-47"

            elif h_level == 'HL4':
                container_cpuconf["worker1"] = "0-8"
                container_cpuconf["worker2"] = "9-16"
                container_cpuconf["worker3"] = "17-22"
                container_cpuconf["worker4"] = "23-47"

            elif h_level == 'HL8':
                container_cpuconf["worker1"] = "0-5"
                container_cpuconf["worker2"] = "6-11"
                container_cpuconf["worker3"] = "12-15"
                container_cpuconf["worker4"] = "16-47"

    def triggerHLadjustment(self, curr_epoch):
        hl = self.hl_routine[curr_epoch]
        container_cpuconf = self.getContainerCPUConfigurations(h_level=hl)
        cpu_set = container_cpuconf[self.worker_name]
        f = open(self.cpu_log, 'w')
        f.write(f'inflating/deflating container {self.worker_name} to cpu_set {cpu_set}\n')
        f.close()
        logging.info(f'changing CPU resource config at epoch {curr_epoch} with containers {container_cpuconf.items()}')

    def getDynamicHLRoutine(self):
        hl_routine = {}
        if self.model_name == 'resnet18' or self.model_name == 'gpt2':
           for e in range(45):
               if 0 <= e < 5:
                   hl_routine[e] = "HL1"
               if 5 <= e < 10:
                   hl_routine[e] = "HL2"
               if 10 <= e < 15:
                   hl_routine[e] = "HL4"
               if 15 <= e < 20:
                   hl_routine[e] = "HL8"
               if 20 <= e < 25:
                   hl_routine[e] = "HL8"
               if 25 <= e < 30:
                   hl_routine[e] = "HL4"
               if 30 <= e < 35:
                   hl_routine[e] = "HL2"
               if 35 <= e < 46:
                   hl_routine[e] = "HL1"

        elif self.model_name == 'resnet50':
            for e in range(200):
                if 0 <= e < 25:
                    hl_routine[e] = "HL1"
                if 25 <= e < 50:
                    hl_routine[e] = "HL2"
                if 50 <= e < 75:
                    hl_routine[e] = "HL4"
                if 75 <= e < 100:
                    hl_routine[e] = "HL8"
                if 100 <= e < 125:
                    hl_routine[e] = "HL8"
                if 125 <= e < 150:
                    hl_routine[e] = "HL4"
                if 150 <= e < 175:
                    hl_routine[e] = "HL2"
                if 175 <= e <= 200:
                    hl_routine[e] = "HL1"

        elif self.model_name == 'alexnet':
            for e in range(160):
                if 0 <= e < 20:
                    hl_routine[e] = "HL1"
                if 20 <= e < 40:
                    hl_routine[e] = "HL2"
                if 40 <= e < 60:
                    hl_routine[e] = "HL4"
                if 60 <= e < 80:
                    hl_routine[e] = "HL8"
                if 80 <= e < 100:
                    hl_routine[e] = "HL8"
                if 100 <= e < 120:
                    hl_routine[e] = "HL4"
                if 120 <= e < 145:
                    hl_routine[e] = "HL2"
                if 140 <= e <= 160:
                    hl_routine[e] = "HL1"

        elif self.model_name == 'vgg11':
            for e in range(35):
                if 0 <= e < 4:
                    hl_routine[e] = "HL1"
                if 4 <= e < 8:
                    hl_routine[e] = "HL2"
                if 8 <= e < 12:
                    hl_routine[e] = "HL4"
                if 12 <= e < 16:
                    hl_routine[e] = "HL8"
                if 16 <= e < 20:
                    hl_routine[e] = "HL8"
                if 20 <= e < 24:
                    hl_routine[e] = "HL4"
                if 24 <= e < 28:
                    hl_routine[e] = "HL2"
                if 28 <= e < 35:
                    hl_routine[e] = "HL1"

        return hl_routine


def toggle_containers():
    dir = os.getcwd()
    cpufiles = ['cpu-0.log', 'cpu-1.log', 'cpu-2.log', 'cpu-3.log']
    while True:
        for cpulog in cpufiles:
            if os.path.isfile(os.path.join(dir, cpulog)):
                time.sleep(1)
                f = open(os.path.join(dir, cpulog), 'r')
                for line in f.readlines():
                    if 'inflating/deflating container' in line:
                        container = line.split()[2]
                        cpuset = line.split()[5]
                        subprocess.run(["docker", "update", "--cpuset-cpus", cpuset, container])
                        print(f'setting docker container {container} to cpuset {cpuset}')
                f.close()
                os.remove(os.path.join(dir, cpulog))

if __name__ == '__main__':
    print('going to track for container triggers for CPU throttling...')
    toggle_containers()