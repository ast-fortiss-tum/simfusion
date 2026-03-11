import roslaunch.parent
import roslaunch.rlutil
import rospy
import simulations.config as config
import time
import subprocess

class LaunchOperations:
    def __init__(self):
        pass

    def clear(self):
        pass
        
    def launch_tools(self):
        if len(launch_args) == 0:
            launch_args = [f"{config.launch['path']}{config.launch['file']}"]
            launch_args.extend(config.launch['args'])
        print(f"launch_args: {launch_args}\n")
        roslaunch_package = 'autoware_mini'
        roslaunch_args =  launch_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(launch_args)[0], roslaunch_args)]


        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        
        self.ros_launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        self.ros_launch.verbose = False
        self.ros_launch.start()

        time.sleep(5)


    def is_process_running(self, process_name):
        """
        Check if a process with a given name is running.
        Returns True if running, False otherwise.
        """
        pids = self.process_pid(process_name)
        if len(pids) > 1:
            return True
        return bool(pids[0])
        
    def process_pid(self, process_name):
        try:
            output = subprocess.check_output(['pgrep', '-x', process_name], text=True)
            pids = [int(pid.strip()) for pid in output.strip().split()]
            print(pids)
            return pids
        except subprocess.CalledProcessError:
            print(f"err: {process_name} is not running")
            return [0]
        

    def start_roslaunch(self, launchfile='start_sim', args=[""]):
        if self.is_process_running('roslaunch'):
            print('roslaunch is already running')
            for pid in self.process_pid('roslaunch'):
                print(f"killing instance {pid} of roslaunch")
                kill(pid, 9)
                time.sleep(2)
    
        print("starting roslaunch...")
        command = ['roslaunch', 'autoware_mini', launchfile+".launch"] # + args #.extends(args)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wait_time = 15
        for _ in range(0, 2):
            print(f"waiting {wait_time}s...")
            time.sleep(wait_time)
            wait_time *= 2/3
            if self.is_process_running('roslaunch'):
                break
        if self.is_process_running('roslaunch'):
                print('roslaunch started successfully.')
        else:
            print('ERR: could not start roslaunch.')
        return process
