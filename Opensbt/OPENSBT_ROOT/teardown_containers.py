import docker
import os

# Connect to Docker daemon
client = docker.from_env()

# Containers to stop and remove
target_containers = {"autoware_mini_lofi", "autoware_mini_hifi", "carla_sim"}

# Get the current container ID to avoid stopping itself (optional)
current_container_id = os.environ.get("HOSTNAME")

# List all containers (running and stopped)
containers = client.containers.list(all=True)

for container in containers:
    if container.name in target_containers:
        # Skip current container just in case its name matches
        if container.id.startswith(current_container_id):
            continue

        print(f"Stopping container: {container.name} ({container.id})")
        try:
            container.stop(timeout=10)
            print(f"Stopped {container.name}")
        except Exception as e:
            print(f"Failed to stop {container.name}: {e}")

        print(f"Removing container: {container.name} ({container.id})")
        try:
            container.remove(force=True)
            print(f"Removed {container.name}")
        except Exception as e:
            print(f"Failed to remove {container.name}: {e}")
