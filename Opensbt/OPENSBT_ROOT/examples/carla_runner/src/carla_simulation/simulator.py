# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla

CAR_NAME = "ego_vehicle"


class Simulator:

    _host = None
    _resolution = None

    def __init__(self, host, port, 
                 timeout, 
                 resolution = 0.24, 
                 rendering = False,
                 max_substep_delta_time = 0.0165,
                 max_substeps = 15):
        
        self._host = host
        self._resolution = resolution
        self._max_substep_delta_time = max_substep_delta_time
        self._max_substeps = max_substeps

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = resolution  
        settings.substepping = True
        settings.max_substep_delta_time = max_substep_delta_time
        settings.max_substeps = max_substeps
        #max substep is the amount of step you can do a s 
        #
        self.world.apply_settings(settings)
        

        traffic_manager = self.client.get_trafficmanager(
            self.get_traffic_manager_port()
        )
        traffic_manager.set_synchronous_mode(True)

    def clean_up(self):
        #settings = self.world.get_settings()
        #settings.synchronous_mode = False
        #settings.fixed_delta_seconds = 0.0
        #self.world.apply_settings(settings)
        actor_list = self.world.get_actors()

        # Destroy all actors safely
        for actor in actor_list:
            if actor.is_alive and actor.attributes.get('role_name') != CAR_NAME:
                actor.destroy()

    def sync_mode_on(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._resolution
        self.world.apply_settings(settings)

    def get_traffic_manager_port(self):
        return 8000 + int(self._host.split('.')[-1])

    def get_client(self):
        return self.client
    
    def get_world(self):
        return self.world
