import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dp import dynaplex
import sys

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1f77b4', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
    '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d',
    '#9edae5' 
]

def visualize_dynamic_world(iter, trace, vars, ax):
    dynamic_state = trace[iter]["state"]
    time = dynamic_state["current_time"]
    task_types = vars["task_types"]
    resource_types = vars["resource_types"]
    ax.clear()  # Clear previous drawings to update with new state

     # Constants for layout
    box_height = 0.6
    box_width = 2.0
    start_y = 5
    padding = 0.5

    # Draw resources and any jobs assigned to them
    for i, (resource_id, resource_info) in enumerate(dynamic_state["resources"].items(), start=1):
        resource_type_key = resource_types[resource_info['resource_type_index']]['key']
        resource_box = patches.Rectangle((i * 3, start_y), box_width, box_height, linewidth=1, edgecolor='black',
                                         facecolor='none')
        ax.add_patch(resource_box)
        ax.text(i * 3 + box_width / 2, start_y + box_height / 2, f"Resource {resource_id}\nType: {resource_type_key}",
                ha='center', va='center')

        # If the resource has an assigned job, show it inside the box
        if "current_assignment_id" in resource_info:
            assignment_id = resource_info["current_assignment_id"]
            job_id = dynamic_state["assignments"][str(assignment_id)]["job_id"]
            job_info = dynamic_state["jobs"][str(job_id)]
            job_task_index = job_info["task_sequence_indices"][0]
            job_task_key = task_types[job_task_index]['key']
            job_number = job_info["job_number"]
            color_index = int(job_number) % 21
            job_color = colors[color_index]

            ax.text(i * 3 + box_width / 2, start_y - box_height, f"Job {job_number}: Task {job_task_key}", ha='center',
                    va='center', fontsize=9, color=job_color)

    # Categorize and display unassigned jobs by task type
    categorized_jobs = {task_type['key']: [] for task_type in task_types}  # Initialize categories
    if dynamic_state["jobs"] is not None:
        for job_id, job_info in dynamic_state["jobs"].items():
            if "current_assignment_id" not in job_info:
                current_task_index = job_info["task_sequence_indices"][0]
                current_task_key = task_types[current_task_index]['key']
                categorized_jobs[current_task_key].append(job_info["job_number"])

    # Initialize the starting x position for the first task type
    start_x = 2
    # Define the offset between different task types
    x_offset = 3.5

    # Visualize categorized jobs
      # Adjust starting y position for categorized jobs
    for task_type, jobs in categorized_jobs.items():
        start_y = 3
        ax.text(start_x, start_y, f"Task {task_type}:", ha='left', va='center', fontsize=9, color='black')
        for job_id in jobs:
            color_index = int(job_id) % 21
            job_color = colors[color_index]
            start_y -= box_height
            ax.text(start_x, start_y, f"Job {job_id}", ha='left', va='center', fontsize=9, color=job_color)
        start_x += x_offset
    # Display the current index or simulation step
    ax.text(1, 9, f"Current Entry: {iter}", fontsize=12, color='black')
    ax.text(1, 8, f"Current time: {time}", fontsize=12, color='black')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')  # Hide the axes

folder_name = "resource_allocation"
mdp_version_number = 1
path_to_json = dynaplex.filepath("mdp_config_examples", folder_name, f"mdp_config_{mdp_version_number}.json")

try:
    with open(path_to_json, "r") as input_file:
        vars = json.load(input_file)    # vars can be initialized manually with something like
except FileNotFoundError:
    raise FileNotFoundError(f"File {path_to_json} not found. Please make sure the file exists and try again.")
except:
    raise Exception("Something went wrong when loading the json file. Have you checked the json file does not contain any comment?")

mdp = dynaplex.get_mdp(**vars)
policy = mdp.get_policy("shortest_processing_time")
demonstrator = dynaplex.get_demonstrator(max_period_count=1000)
trace = demonstrator.get_trace(mdp,policy)

#print(vars)
#sys.exit()

#for frame in trace:
#    print(frame[ "state"])

# Setup figure and axes for the animation
fig, ax = plt.subplots(figsize=(10, 6))

# Create the animation
ani = FuncAnimation(fig, visualize_dynamic_world, frames=range(1, 500), fargs=(trace, vars, ax), repeat=False)

#ani.save(dynaplex.filepath("movies","animation.gif"), writer='Pillow', fps=2)

plt.show()