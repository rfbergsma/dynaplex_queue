import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from datetime import timedelta
from dp import dynaplex

def draw_static_background_with_labels(ax, vars):
    vehicle_graph = vars['vehicle_graph']
    rows = vehicle_graph['rows']
    pick_locations = vars['pick_locations']

    num_columns = len(rows[0].split('|'))
    num_rows = len(rows)

    # Set axis limits slightly larger to accommodate labels
    ax.set_xlim(-1, num_columns)
    ax.set_ylim(-1, num_rows)

    light_rosa_pink = '#fff0f5'

    # Draw the grid and directions
    for y, row in enumerate(rows):
        cells = row.split('|')
        for x, cell in enumerate(cells):
            facecolor = 'none' if cell.strip() else 'black'
            rect = patches.Rectangle((x, num_rows - y - 1), 1, 1, linewidth=1, edgecolor='grey', facecolor=facecolor)
            ax.add_patch(rect)

            # Draw directions within each cell
            for direction in cell.strip():
                dx, dy = 0, 0
                if direction == 'L':
                    dx = -0.5
                elif direction == 'R':
                    dx = 0.5
                if direction == 'U':
                    dy = 0.5
                elif direction == 'D':
                    dy = -0.5
                if dx or dy:
                    ax.arrow(x + 0.5, num_rows - y - 0.5, dx * 0.4, dy * 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    # Highlight pick locations with light rosa pink color
    for location in pick_locations:
        y, x = location['coords']
        rect = patches.Rectangle((x, num_rows - y - 1), 1, 1, linewidth=1, edgecolor='grey', facecolor=light_rosa_pink)
        ax.add_patch(rect)

    # Label columns at the top
    for x in range(num_columns):
        if x % 5 == 0 or x == num_columns - 1:  # Label multiples of 5 and the last one
            ax.text(x + 0.5, num_rows, str(x), ha='center', va='bottom', fontsize=10, color='black')

    # Label rows on the left side
    for y in range(num_rows):
        if y % 5 == 0 or y == num_rows - 1:  # Label multiples of 5 and the last one
            ax.text(-0.5, num_rows - y - 0.5, str(y), ha='right', va='center', fontsize=10, color='black')

    plt.axis('off')


colors = [
    '#1f77b4',  # Vivid blue
    '#ff7f0e',  # Bright orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brownish
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive green
    '#17becf',  # Sky blue
    '#aec7e8',  # Light blue
    '#ffbb78',  # Peach
    '#98df8a',  # Pale green
    '#ff9896',  # Soft red
    '#c5b0d5',  # Lilac
    '#c49c94',  # Dusty pink
    '#f7b6d2',  # Pastel pink
    '#c7c7c7',  # Light gray
    '#dbdb8d',  # Light olive
    '#9edae5',  # Pale cyan
    '#393b79',  # Dark blue
    '#5254a3',  # Periwinkle
    '#6b6ecf',  # Soft blue
    '#9c9ede',  # Lavender blue
    '#637939',  # Olive drab
    '#8ca252',  # Muted green
    '#b5cf6b',  # Pear
    '#cedb9c',  # Light lime
    '#8c6d31',  # Bronze
    '#bd9e39',  # Goldish
    '#e7ba52'   # Saffron
]

dynamic_elements = []


def format_time_from_tenths(time_in_tenths):
    # Convert time_in_tenths to seconds
    time_in_seconds = time_in_tenths / 10

    # Convert seconds to timedelta for easy formatting
    time_delta = timedelta(seconds=time_in_seconds)

    # For times less than a day, format hours, minutes, seconds, and milliseconds
    formatted_time = "{:02}:{:02}:{:06.3f}".format(time_delta.seconds // 3600, (time_delta.seconds % 3600) // 60,
                                                   time_delta.seconds % 60 + time_delta.microseconds / 1e6)

    return formatted_time
def visualize_dynamic_world(iter, trace, vars, ax):
    global dynamic_elements  # Ensure you're using the global variable

    # Clear previous dynamic elements
    for element in dynamic_elements:
        element.remove()
    dynamic_elements.clear()

    dynamic_state = trace[iter]["state"]
    time = dynamic_state["current_time"]
    #can I somehow print the time somewhere? It seems all space is taken by the grid allready
    vehicles = dynamic_state["vehicles"]
    pickers = dynamic_state["pickers"]
    grid_width = len(vars['vehicle_graph']['rows'][0].split('|'))
    num_rows = len(vars['vehicle_graph']['rows'])

    # Place the time text just above the axes
    text = fig.text(0.5, 0.95, f"Time: {format_time_from_tenths(time)}  Frame:{iter}", ha='center', va='center', fontsize=10)
    dynamic_elements.append(text)
    #print(dynamic_state)
    # First, draw vehicles
    vehicle_id_range = range(0, 1)
    if vehicles:
        for vehicle_id, vehicle_info in vehicles.items():
            if 'node_id' in vehicle_info:
                node_id = vehicle_info['node_id']
                color_index = int(vehicle_id) % len(colors)
                color = colors[color_index]

                # Convert node_id to (x, y) coordinates
                y, x = divmod(node_id, grid_width)
                y = num_rows - y - 1  # Adjust y to match the drawn grid

                size = 0.4

                # Draw the vehicle
                rect = patches.Rectangle((x + size / 2, y + size / 2), size, size, linewidth=3, edgecolor=color,
                                         facecolor=color)
                ax.add_patch(rect)
                dynamic_elements.append(rect)
                vehicle_id_int = int(vehicle_id)
                if vehicle_id_int in vehicle_id_range:  # Adjust this condition for other ranges
                    if 'remaining_pick_list' in vehicle_info:
                        for pick_node_id in vehicle_info.get('remaining_pick_list', []):
                            py, px = divmod(pick_node_id, grid_width)
                            py = num_rows - py - 1  # Adjust y to match the drawn grid

                            # Draw a small dot at the pick location
                            pick_dot = patches.Circle((px + 0.8, py + 0.8), 0.1, color=color, fill=True)
                            ax.add_patch(pick_dot)
                            dynamic_elements.append(pick_dot)
                    if "drop_off_node_id" in vehicle_info:
                        drop_off_id = vehicle_info.get("drop_off_node_id")
                        py, px = divmod(drop_off_id, grid_width)
                        py = num_rows - py - 1  # Adjust y to match the drawn grid

                        # Draw a small dot at the pick location
                        pick_dot = patches.Circle((px + 0.2, py + 0.2), 0.1, color=color, fill=True)
                        ax.add_patch(pick_dot)
                        dynamic_elements.append(pick_dot)

    # Then, draw pickers as numbered circles
    for picker_id, picker_info in pickers.items():
        node_id = picker_info['node_id']
        y, x = divmod(node_id, grid_width)
        y = num_rows - y - 1  # Adjust y to match the drawn grid

        # Draw the picker circle
        circle = patches.Circle((x + 0.2, y + 0.2), 0.15, color='black', fill=True)
        ax.add_patch(circle)
        dynamic_elements.append(circle)

        # Add picker_id as text in the circle
        text = ax.text(x + 0.2, y + 0.2, str(picker_id), color='white', ha='center', va='center', fontsize=8)
        dynamic_elements.append(text)

        # If the picker is assigned to a vehicle, show their number over the vehicle
        if 'assigned_vehicle_id' in picker_info:
            # Find the vehicle's position for the assigned picker
            assigned_vehicle_id = picker_info['assigned_vehicle_id']
            vehicle_info = vehicles.get(str(assigned_vehicle_id))  # Vehicle IDs might be strings
            if vehicle_info and 'node_id' in vehicle_info:
                vehicle_node_id = vehicle_info['node_id']
                vy, vx = divmod(vehicle_node_id, grid_width)
                vy = num_rows - vy - 1

                # Overlay picker_id text on the vehicle
                text = ax.text(vx + 0.5, vy + 0.5, str(picker_id), color='white', ha='center', va='center', fontsize=8)
                dynamic_elements.append(text)


folder_name = "collaborative_picking"
mdp_version_number = 3 #3
path_to_json = dynaplex.filepath("mdp_config_examples", folder_name, f"mdp_config_{mdp_version_number}.json")

try:
    with open(path_to_json, "r") as input_file:
        vars = json.load(input_file)  # vars can be initialized manually with something like
        vars["graph_feats"] = False
except FileNotFoundError:
    raise FileNotFoundError(f"File {path_to_json} not found. Please make sure the file exists and try again.")
except:
    raise Exception(
        "Something went wrong when loading the json file. Have you checked the json file does not contain any comment?")
# vars["timestep_delta"] = 20
# vars["startup_duration"] = 12000
mdp = dynaplex.get_mdp(**vars)
policy = mdp.get_policy("closest_pair")
# demonstrator = dynaplex.get_demonstrator(max_period_count=3000, restricted_statecategory={"await": "event", "index": 0})
demonstrator = dynaplex.get_demonstrator(max_period_count=1000, restricted_statecategory={"await": "event", "index": 0})
trace = demonstrator.get_trace(mdp, policy)

# Setup figure and axes for the animation
fig, ax = plt.subplots(figsize=(10, 6))
draw_static_background_with_labels(ax, mdp.get_static_info()['config_info'])


#plt.show()
#exit()

# Create the animation
# ani = FuncAnimation(fig, visualize_dynamic_world, frames=range(2700, 2900), interval=2, fargs=(trace, mdp.get_static_info()['config_info'], ax),repeat=False)
ani = FuncAnimation(fig, visualize_dynamic_world, frames=range(500, 700), interval=2, fargs=(trace, mdp.get_static_info()['config_info'], ax),repeat=False)

#ani.save(dynaplex.filepath("movies","animation.gif"), writer='Pillow', fps=2)

plt.show()
