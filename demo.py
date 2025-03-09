import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Define the parent-child relationships for the keypoints
s_coco_parent_ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14], dtype=int)

def draw_keypoints(ax, keypoints_3d, color):
    # Plot each keypoint
    for i in range(keypoints_3d.shape[0]):
        ax.scatter(keypoints_3d[i, 0], keypoints_3d[i, 1], keypoints_3d[i, 2], c=color, marker='o', s=1, zorder=4)

    # Connect the keypoints based on the parent-child relationship
    for child_id in range(1, len(s_coco_parent_ids)):
        parent_id = s_coco_parent_ids[child_id]
        
        # Plot a line between the parent and child
        ax.plot([keypoints_3d[parent_id, 0], keypoints_3d[child_id, 0]],
                [keypoints_3d[parent_id, 1], keypoints_3d[child_id, 1]],
                [keypoints_3d[parent_id, 2], keypoints_3d[child_id, 2]], 'k-', lw=1, zorder=3)

def draw_background(ax, width, length, paint_length, paint_width):
    x = np.linspace(- width / 2, paint_length - width / 2, 10)
    y = np.linspace((length - paint_width)/2 - length / 2, (length + paint_width)/2 - length / 2, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, color='#DCDCDC', alpha=1, zorder=1)

    x = np.linspace(-width/2, width/2, 10)
    y = np.linspace(-length/2, length/2, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, color='#fafafa', alpha=0.5, zorder=1)


def draw_box(ax, court_width, court_length, court_paint_length, court_paint_width):
    ax.plot([- court_width / 2, - court_width / 2], [- court_length / 2, court_length / 2], [0, 0], 'k-', lw=1, zorder=2)
    ax.plot([court_width / 2 , court_width / 2], [- court_length / 2, court_length / 2], [0, 0], 'k-', lw=1, zorder=2)
    ax.plot([- court_width / 2, court_width / 2], [- court_length / 2, - court_length / 2], [0, 0], 'k-', lw=1, zorder=2)
    ax.plot([- court_width / 2, court_width / 2], [court_length / 2, court_length / 2], [0, 0], 'k-', lw=1, zorder=2)

    ax.plot([- court_width / 2, court_paint_length - court_width / 2], [(court_length - court_paint_width)/2 - court_length / 2, (court_length - court_paint_width)/2 - court_length / 2], [0, 0], 'k-', lw=1, zorder=2)
    ax.plot([court_paint_length  - court_width / 2, court_paint_length - court_width / 2], [(court_length - court_paint_width)/2 - court_length / 2, (court_length + court_paint_width)/2 - court_length / 2], [0, 0], 'k-', lw=1, zorder=2)
    ax.plot([court_paint_length - court_width / 2, 0 - court_width / 2], [(court_length + court_paint_width)/2 - court_length / 2, (court_length + court_paint_width)/2 - court_length / 2], [0, 0], 'k-', lw=1, zorder=2)

def compute_view_angles(camera_pos):
    azim = np.degrees(np.arctan2(camera_pos[1], camera_pos[0]))
    elev = np.degrees(np.arctan2(camera_pos[2], np.sqrt(camera_pos[0]**2 + camera_pos[1]**2)))
    return elev, azim

def draw_arcs(ax, court_length, court_width, paint_length, paint_width):
    # 3 point line
    basket_loc = np.array([1.575, court_length / 2, 0])
    basket_loc[0] -= court_width / 2
    basket_loc[1] -= court_length / 2

    radius = 6.75
    corner_dis = 0.9
    angle = np.arccos((court_length / 2 - corner_dis) / radius)

    theta = np.linspace(angle, np.pi - angle, 100)

    y = radius * np.cos(theta)
    x = radius * np.sin(theta)

    ax.plot(x + basket_loc[0], y + basket_loc[1], np.zeros_like(x), color='black', label="3-Point Line", lw=1, zorder=2)

    l0 = radius * np.sin(angle) + (basket_loc[0] + court_width / 2)
    ax.plot([-court_width / 2, l0 - court_width / 2], [corner_dis - court_length / 2, corner_dis - court_length / 2], [0, 0], color='black', lw=1, zorder=2)
    ax.plot([-court_width / 2, l0 - court_width / 2], [court_length / 2 - corner_dis, court_length / 2 - corner_dis], [0, 0], color='black', lw=1, zorder=2)

    # no-charge semi-circle line
    basket_loc = np.array([1.575, court_length / 2, 0])
    basket_loc[0] -= court_width / 2
    basket_loc[1] -= court_length / 2

    radius = 1.25
    corner_dis = 0.9
    theta = np.linspace(0, np.pi, 100)

    y = radius * np.cos(theta)
    x = radius * np.sin(theta)

    ax.plot(x + basket_loc[0], y + basket_loc[1], np.zeros_like(x), color='black', label="3-Point Line", lw=1, zorder=2)

    l0 = 0.375
    ax.plot([basket_loc[0], basket_loc[0] - l0], [-radius, -radius], [0, 0], color='black', lw=1, zorder=2)
    ax.plot([basket_loc[0], basket_loc[0] - l0], [radius, radius], [0, 0], color='black', lw=1, zorder=2)

    # free throw line
    radius = 1.8
    theta = np.linspace(0, np.pi, 100)

    y = radius * np.cos(theta)
    x = radius * np.sin(theta)

    ax.plot(x + paint_length - court_width / 2, y, np.zeros_like(x), color='black', label="Free throw Line", lw=1, zorder=2)

def draw_painted_bars(ax, court_length, court_width, paint_length, paint_width):
    y0 = (court_length - paint_width) / 2
    y1 = (court_length + paint_width) / 2

    bar_len = 0.1
    bar_loc = [1.75, 1.75 + 0.85 + 0.4 + 0.85, 1.75 + 0.85 + 0.4 + 0.85 + 0.85]
    for b_l in bar_loc:
        ax.plot([b_l - court_width / 2, b_l - court_width / 2], [y0 - court_length / 2, y0 - bar_len - court_length / 2], [0, 0], color='black', lw=1, zorder=2)
        ax.plot([b_l - court_width / 2, b_l - court_width / 2], [y1 - court_length / 2, y1 + bar_len - court_length / 2], [0, 0], color='black', lw=1, zorder=2)

    bar_loc2 = 1.75 + 0.85 
    bar_width = 0.4
    x = np.linspace(bar_loc2 - court_width / 2, bar_loc2 + bar_width - court_width / 2, 10)
    y = np.linspace(y0 - bar_len - court_length / 2, y0 - court_length / 2, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, color='black', alpha=1, zorder=1)
    y = np.linspace(y1 + bar_len - court_length / 2, y1 - court_length / 2, 10)
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, Z, color='black', alpha=1, zorder=1)

def draw_board(ax, court_length, court_width):
    board_length = 1.05
    board_width = 1.8
    board_loc = [1.2, 0, 2.9]
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [-board_width / 2, -board_width / 2], [board_loc[2], board_loc[2] + board_length], color='black', lw=1, zorder=2)
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [-board_width / 2, board_width / 2], [board_loc[2], board_loc[2]], color='black', lw=1, zorder=2)
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [board_width / 2, board_width / 2], [board_loc[2], board_loc[2] + board_length], color='black', lw=1, zorder=2)
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [-board_width / 2, board_width / 2], [board_loc[2] + board_length, board_loc[2] + board_length], color='black', lw=1, zorder=2)

    board_length = 0.45
    board_width = 0.59
    board_loc = [1.2, 0, 2.9 + 0.15]
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [-board_width / 2, -board_width / 2], [board_loc[2], board_loc[2] + board_length], color='black', lw=1, zorder=2)
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [-board_width / 2, board_width / 2], [board_loc[2], board_loc[2]], color='black', lw=1, zorder=2)
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [board_width / 2, board_width / 2], [board_loc[2], board_loc[2] + board_length], color='black', lw=1, zorder=2)
    ax.plot([board_loc[0] -court_width / 2, board_loc[0] -court_width / 2], [-board_width / 2, board_width / 2], [board_loc[2] + board_length, board_loc[2] + board_length], color='black', lw=1, zorder=2)


def vis_frame(keypoints, frame_id, output_file='temp.png', vis_frame_id=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # set range
    offset = 1
    court_length = 15.05
    court_width = 11.05
    court_paint_width = 4.9
    court_paint_length = 5.8

    ax.set_xlim([- court_width / 2 - offset, court_width / 2  + offset])
    ax.set_ylim([- court_length / 2  - offset, court_length / 2  + offset])
    ax.set_zlim([0, 3.5])

    # elev, azim = compute_view_angles([1, -0.5, 0.2])
    ax.view_init(elev=19, azim=-30)

    # draw background
    draw_background(ax, court_width, court_length, court_paint_length, court_paint_width)

    player_list = list(keypoints.keys())
    host_list = sorted([player for player in player_list if 'host' in player])
    guest_list = sorted([player for player in player_list if 'guest' in player])

    host_dict = {player: i for i, player in enumerate(host_list)}
    guest_dict = {player: i for i, player in enumerate(guest_list)}

    host_color = {
        0: '#ff6347',
        1: '#ff4500',
        2: '#ffa500',
    }
    guest_color = {
        0: '#1e90ff',
        1: '#00bfff',
        2: '#00ced1',
    }

    for player, keypoints_3d in keypoints.items():
        kp_3d = keypoints_3d[frame_id].copy()
        kp_3d[..., 0] -= court_width / 2
        kp_3d[..., 1] -= court_length / 2
        color = host_color[host_dict[player]] if 'host' in player else guest_color[guest_dict[player]]

        draw_keypoints(ax, kp_3d, color)

    # draw court
    draw_box(ax, court_width, court_length, court_paint_length, court_paint_width)
    draw_arcs(ax, court_length, court_width, court_paint_length, court_paint_width)
    draw_painted_bars(ax, court_length, court_width, court_paint_length, court_paint_width)
    draw_board(ax, court_length, court_width)

    # set equal aspect ratio
    ax.set_aspect('equal')
    # remove grid
    ax.grid(False)
    # remove background
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # remove axis
    ax._axis3don = False

    fig.tight_layout()

    # canvas = FigureCanvas(fig)
    # canvas.draw()

    # width, height = fig.get_size_inches() * fig.get_dpi()

    # img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.savefig(output_file, dpi=800)

    # crop center
    scale = 2.6667
    cropped_length = int(1140 * scale)
    cropped_width = int(504 * scale)

    img = cv2.imread(output_file)
    img_width, img_length = img.shape[:2]

    start_x = (img_width - cropped_width) // 2 + int(25 * scale)
    start_y = (img_length - cropped_length) // 2 + int(22 * scale)
    img = img[start_x:start_x + cropped_width, start_y:start_y + cropped_length]

    # write frame idx on img
    if vis_frame_id:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 100)
        fontScale = 2
        fontColor = (0, 0, 0)
        lineType = 5
        img = cv2.putText(img, 'Frame: {}'.format(frame_id), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    plt.close(fig)
    return img


if __name__ == '__main__':
    # keypoint coordinate:
    # top-left corner as (0, 0, 0)  half-court width as x-axis half-court length as y-axis
    # vis coordinate:
    # center of the court as (0, 0, 0) half-court width as x-axis half-court length as y-axis
    keypoints = np.load('data/basketball/joints/S4_09_000212_000219_pose.npy', allow_pickle=True).item()
    frame_id = 0
    output_file = 'temp.png'

    img = vis_frame(keypoints, frame_id, vis_frame_id=False)
    cv2.imwrite(f'temp.png', img)