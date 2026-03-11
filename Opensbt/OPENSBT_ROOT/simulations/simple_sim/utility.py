import math
import xml.etree.ElementTree as ET

CAR_NAME = "ego_vehicle"

class Utility(object):
    @staticmethod
    def calculate_magnitude(x, y, z):
        """
        Computes the magnitude of a 3D vector.
        """
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def calculate_acceleration(prev_velocity, curr_velocity, duration):
        """
        Computes the acceleration given two velocity vectors and a time duration.
        """
        acc_vector = (
            (curr_velocity[0] - prev_velocity[0]) / duration,
            (curr_velocity[1] - prev_velocity[1]) / duration,
            (curr_velocity[2] - prev_velocity[2]) / duration
        )
        return Utility.calculate_magnitude(acc_vector[0], acc_vector[1], acc_vector[2])
    
    @staticmethod
    def extract_timestamp(sec, nsec=0):
        """
        Extracts the timestamp from a ROS message.
        """
        return sec + (nsec / 1e9)
    
    @staticmethod
    def extract_info_from_scenario_file(xml_content):
        try:
            with open(xml_content, "r", encoding="utf-8") as file:
                xml_content = file.read()

            # Check XML content
            xml_content = xml_content.strip()
            if not xml_content:
                raise ValueError("Error: Empty XML Content!")

            # XML parsing
            root = ET.fromstring(xml_content)

            vehicle_info = {}
            adversary_info = {}
            parameter_values = {}

            # ParameterDeclarations information
            parameters = root.find("ParameterDeclarations")
            if parameters is not None:
                for parameter in parameters.findall("ParameterDeclaration"):
                    name = parameter.attrib.get("name")
                    value = parameter.attrib.get("value")
                    if name and value:
                        # Dönüştürülebiliyorsa float olarak ata, aksi halde hata verecektir.
                        parameter_values[name] = float(value)



            # ego_vehicle information
            teleport_action = root.find(f".//Private[@entityRef='{CAR_NAME}']//TeleportAction//WorldPosition")
            if teleport_action is not None:
                vehicle_info["position"] = {
                    "x": float(teleport_action.get("x")),
                    "y": float(teleport_action.get("y")),
                    "z": float(teleport_action.get("z")),
                    "h": float(teleport_action.get("h"))
                }
            teleport_action = root.find(f".//Private[@entityRef='adversary']//TeleportAction//WorldPosition")
            if teleport_action is not None:
                adversary_info["position"] = {
                    "x": float(teleport_action.get("x")),
                    "y": float(teleport_action.get("y")),
                    "z": float(teleport_action.get("z")),
                    "h": float(teleport_action.get("h"))
                }

            # start_trigger = root.find(".//Maneuver[@name='PedestrianCrossingManeuver']//StartTrigger")
            # if start_trigger is not None:
            #     reach_position_condition = start_trigger.find(".//ReachPositionCondition")
            #     if reach_position_condition is not None:
            #         position = reach_position_condition.find("Position/WorldPosition")
            #         if position is not None:
            #             adversary_info["position"] = {
            #                 "x": float(position.get("x")),
            #                 "y": float(position.get("y")),
            #                 "z": float(position.get("z")),
            #                 "h": float(position.get("h"))
            #             }

            return {"ego_vehicle": vehicle_info, "adversary": adversary_info, "parameters": parameter_values}

        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")
            return None
        except ValueError as ve:
            print(f"Error: {ve}")
            return None
