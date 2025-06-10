import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # For drawing 3D polygons (cuboids)
import random
import math

# --- Configuration ---

# Define Item Types (id, name, dimensions (length, width, height), color (R, G, B, Alpha))
ITEM_DEFINITIONS = [
    {'id': 1, 'name': 'BoxA', 'dims': (4, 3, 2), 'color': (1, 0, 0, 0.85)},  # Red
    {'id': 2, 'name': 'BoxB', 'dims': (3, 3, 3), 'color': (0, 1, 0, 0.85)},  # Green
    {'id': 3, 'name': 'BoxC', 'dims': (2, 2, 4), 'color': (0, 0, 1, 0.85)},  # Blue
    {'id': 4, 'name': 'BoxD', 'dims': (5, 2, 2), 'color': (1, 1, 0, 0.85)},  # Yellow
    {'id': 5, 'name': 'BoxE', 'dims': (1, 1, 5), 'color': (0, 1, 1, 0.85)},  # Cyan
    {'id': 6, 'name': 'BoxF', 'dims': (2, 1, 3), 'color': (1, 0, 1, 0.85)},  # Magenta
    {'id': 7, 'name': 'BoxG', 'dims': (3, 2, 1), 'color': (0.5, 0.5, 0, 0.85)}, # Olive
    {'id': 8, 'name': 'BoxH', 'dims': (1, 4, 2), 'color': (0, 0.5, 0.5, 0.85)}, # Teal
]

# Define Container Schemas: A list of dictionaries
TRAILER_SCHEMAS = [
    {'id': 'C1_Std', 'dims': (10, 8, 6)},
    {'id': 'C2_Small', 'dims': (7, 6, 5)}, 
    {'id': 'C3_Tall',  'dims': (5, 5, 9)}, 
    {'id': 'C4_Long',  'dims': (12, 4, 4)}, 
]

# Define a sequence of items to pack for an episode (by name) - NOW 12 ITEMS
ITEMS_TO_PACK_SEQUENCE = [
    'BoxA', 'BoxB', 'BoxC', 'BoxD', 'BoxE', 'BoxF', 
    'BoxG', 'BoxH', 'BoxA', 'BoxC', 'BoxE', 'BoxB' 
] 

# Environment Parameters
MAX_INVALID_ATTEMPTS_PER_ITEM = 15 
# Adjust MAX_EPISODE_STEPS based on the new number of items
MAX_EPISODE_STEPS = len(ITEMS_TO_PACK_SEQUENCE) * (MAX_INVALID_ATTEMPTS_PER_ITEM + 1) + (len(TRAILER_SCHEMAS) * 5)
# Calculation: 12 * 16 + 4 * 5 = 192 + 20 = 212

# --- Helper functions for rendering ---
def draw_cuboid(ax, position, dimensions, face_color=(0,0,1,0.5), edge_color='k', linewidth=0.5):
    px, py, pz = position
    dx, dy, dz = dimensions
    vertices = [ (px, py, pz), (px + dx, py, pz), (px + dx, py + dy, pz), (px, py + dy, pz),
                 (px, py, pz + dz), (px + dx, py, pz + dz), (px + dx, py + dy, pz + dz), (px, py + dy, pz + dz) ]
    faces = [ [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
              [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
              [vertices[0], vertices[3], vertices[7], vertices[4]], [vertices[1], vertices[2], vertices[6], vertices[5]] ]
    poly3d = Poly3DCollection(faces, facecolors=face_color, linewidths=linewidth, edgecolors=edge_color, alpha=face_color[3] if len(face_color)==4 else 1.0)
    ax.add_collection3d(poly3d)

def draw_container_wireframe(ax, dims, color='gray', linestyle=':', linewidth=1.0):
    d_x, d_y, d_z = dims
    v = np.array([[0,0,0], [d_x,0,0], [d_x,d_y,0], [0,d_y,0], [0,0,d_z], [d_x,0,d_z], [d_x,d_y,d_z], [0,d_y,d_z]])
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot([v[i,0],v[j,0]], [v[i,1],v[j,1]], [v[i,2],v[j,2]], color=color, linestyle=linestyle, linewidth=linewidth)

class PackingEnv:
    def __init__(self, container_schemas, item_definitions, item_sequence_names):
        self.container_schemas = container_schemas
        self.num_containers = len(container_schemas)
        self.item_definitions = {item['name']: item for item in item_definitions}
        self.item_sequence_names = item_sequence_names
        self.id_to_color_map = {0: (0,0,0,0)}
        for item_def in item_definitions: self.id_to_color_map[item_def['id']] = item_def['color']
        self.num_orientations = 6
        self.reset()

    def _get_oriented_dims(self, base_dims, o_idx): # Shortened var name
        l,w,h=base_dims
        if o_idx==0: return(l,w,h)
        if o_idx==1: return(l,h,w)
        if o_idx==2: return(w,l,h)
        if o_idx==3: return(w,h,l)
        if o_idx==4: return(h,l,w)
        if o_idx==5: return(h,w,l)
        raise ValueError(f"Invalid orientation: {o_idx}")

    def _get_current_item_details(self):
        if self.current_item_idx_to_pack >= len(self.items_to_pack_this_episode): return None
        return self.items_to_pack_this_episode[self.current_item_idx_to_pack]

    def _get_observation(self):
        item=self._get_current_item_details()
        item_info={'id':item['id'],'base_dims':item['dims']} if item else None
        return ([s.copy() for s in self.container_spaces], item_info) # s for space

    def reset(self):
        self.container_spaces = []
        self.container_dims_list = []
        for sch in self.container_schemas: # sch for schema
            dims=np.array(sch['dims'],dtype=int)
            self.container_spaces.append(np.zeros(dims,dtype=int))
            self.container_dims_list.append(dims)
        self.packed_items_info=[]
        self.items_to_pack_this_episode=[]
        for name in self.item_sequence_names:
            if name in self.item_definitions: self.items_to_pack_this_episode.append(self.item_definitions[name].copy())
            else: print(f"Warn: Item '{name}' not found.")
        self.current_item_idx_to_pack=0
        self.total_steps_taken_episode=0
        self.invalid_attempts_current_item=0
        self.total_packed_volume=0
        self.packed_volume_per_container=np.zeros(self.num_containers)
        return self._get_observation()

    def _check_placement(self, c_idx, i_dims_o, pos): # c_idx, item_dims_oriented, position
        if not (0<=c_idx<self.num_containers): return False
        space=self.container_spaces[c_idx]; c_dims=self.container_dims_list[c_idx]
        px,py,pz=np.array(pos,dtype=int); dx,dy,dz=np.array(i_dims_o,dtype=int)
        if not (px>=0 and px+dx<=c_dims[0] and py>=0 and py+dy<=c_dims[1] and pz>=0 and pz+dz<=c_dims[2]): return False
        if np.any(space[px:px+dx, py:py+dy, pz:pz+dz]!=0): return False
        return True

    def step(self, action):
        self.total_steps_taken_episode+=1
        done=False; reward=0.0
        info={'packed_volume_current_item':0,'skipped_item':False,'packed_item_name':None,'packed_in_container_idx':None,'error':None}
        if self.current_item_idx_to_pack>=len(self.items_to_pack_this_episode):
            done=True; return self._get_observation(),reward,done,info
        
        item_def=self.items_to_pack_this_episode[self.current_item_idx_to_pack]
        c_idx,px,py,pz,o_idx=action # container_idx, pos_x, pos_y, pos_z, orientation_idx
        item_base_dims=item_def['dims']; item_id=item_def['id']; item_name=item_def['name']

        if not (0<=c_idx<self.num_containers):
            reward=-3.0; self.invalid_attempts_current_item+=1; info['error']=f"Invalid c_idx: {c_idx}"
            if self.invalid_attempts_current_item>MAX_INVALID_ATTEMPTS_PER_ITEM:
                self.current_item_idx_to_pack+=1; self.invalid_attempts_current_item=0; reward-=5.0; info['skipped_item']=True
                if self.current_item_idx_to_pack>=len(self.items_to_pack_this_episode): done=True
            return self._get_observation(),reward,done,info
        try:
            oriented_dims=self._get_oriented_dims(item_base_dims,o_idx)
        except ValueError as e:
            reward=-2.0; self.invalid_attempts_current_item+=1; info['error']=str(e)
            if self.invalid_attempts_current_item>MAX_INVALID_ATTEMPTS_PER_ITEM:
                self.current_item_idx_to_pack+=1; self.invalid_attempts_current_item=0; reward-=5.0; info['skipped_item']=True
                if self.current_item_idx_to_pack>=len(self.items_to_pack_this_episode): done=True
            return self._get_observation(),reward,done,info

        if self._check_placement(c_idx,oriented_dims,(px,py,pz)):
            dx,dy,dz=oriented_dims
            self.container_spaces[c_idx][px:px+dx,py:py+dy,pz:pz+dz]=item_id
            item_vol=np.prod(oriented_dims); reward=float(item_vol)
            self.total_packed_volume+=item_vol; self.packed_volume_per_container[c_idx]+=item_vol
            self.packed_items_info.append({'name':item_name,'id':item_id,'container_idx':c_idx,'pos':(px,py,pz),'oriented_dims':oriented_dims,'volume':item_vol})
            info.update({'packed_volume_current_item':item_vol,'packed_item_name':item_name,'packed_in_container_idx':c_idx})
            self.current_item_idx_to_pack+=1; self.invalid_attempts_current_item=0
            if self.current_item_idx_to_pack>=len(self.items_to_pack_this_episode): done=True
        else:
            reward=-0.5; self.invalid_attempts_current_item+=1
            if self.invalid_attempts_current_item>MAX_INVALID_ATTEMPTS_PER_ITEM:
                self.current_item_idx_to_pack+=1; self.invalid_attempts_current_item=0; reward-=5.0; info['skipped_item']=True
                if self.current_item_idx_to_pack>=len(self.items_to_pack_this_episode): done=True
        if self.total_steps_taken_episode>=MAX_EPISODE_STEPS and not done:
            done=True; reward-=10.0
        info.update({'total_packed_volume_all_containers':self.total_packed_volume,'packed_volume_per_container':self.packed_volume_per_container.tolist(),
                     'items_processed_count':self.current_item_idx_to_pack,'total_steps_episode':self.total_steps_taken_episode})
        return self._get_observation(),reward,done,info

    def render(self, fig=None, title_prefix="Container Packing"):
        if self.num_containers == 0: return

        cols = math.ceil(math.sqrt(self.num_containers))
        rows = math.ceil(self.num_containers / cols)

        subplot_w, subplot_h = 5, 4.5 
        if fig is None:
            fig = plt.figure(figsize=(cols * subplot_w, rows * subplot_h))
        fig.clear() 
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.92, wspace=0.15, hspace=0.25)

        for i in range(self.num_containers):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            c_dims_i = self.container_dims_list[i] 
            c_id_i = self.container_schemas[i]['id'] 
            draw_container_wireframe(ax, c_dims_i, color='dimgray', linestyle='-', linewidth=1.0)
            items_count = 0
            for item_info in self.packed_items_info:
                if item_info['container_idx'] == i:
                    items_count+=1
                    item_color = self.id_to_color_map.get(item_info['id'], (0.5,0.5,0.5,0.7))
                    draw_cuboid(ax, item_info['pos'], item_info['oriented_dims'], face_color=item_color, edge_color='black', linewidth=0.6)
            vol_i = self.packed_volume_per_container[i]; total_vol_i = np.prod(c_dims_i)
            util_i = (vol_i/total_vol_i)*100 if total_vol_i>0 else 0
            ax.set_xlabel('X', fontsize=9); ax.set_ylabel('Y', fontsize=9); ax.set_zlabel('Z', fontsize=9)
            ax.set_title(f"{c_id_i}\nItems:{items_count}, Vol:{vol_i:.0f}, Util:{util_i:.1f}%", fontsize=10, pad=-2)
            ax.set_xlim([0, c_dims_i[0]]); ax.set_ylim([0, c_dims_i[1]]); ax.set_zlim([0, c_dims_i[2]])
            ax.view_init(elev=25., azim=-70) 
            try:
                ax.set_box_aspect(c_dims_i)
            except AttributeError: 
                max_dim = np.max(c_dims_i)
                ax.auto_scale_xyz([0,max_dim],[0,max_dim],[0,max_dim])
            ax.tick_params(axis='both', which='major', labelsize=7, pad=1)
            ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0)); ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0)); ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
            ax.grid(False)
        fig.suptitle(title_prefix, fontsize=14, y=0.99) 
        if fig.get_axes() and plt.isinteractive(): plt.draw() 
        return fig

def run_random_agent_episode(env, render_each_step=False, final_render=True):
    obs = env.reset()
    done=False; total_reward=0.0; step_count=0
    fig_render=None
    cols_sim = math.ceil(math.sqrt(env.num_containers))
    rows_sim = math.ceil(env.num_containers / cols_sim)
    sim_fig_width = cols_sim * 5 
    sim_fig_height = rows_sim * 4.5 

    if render_each_step: 
        plt.ion()
        fig_render = plt.figure(figsize=(sim_fig_width, sim_fig_height))

    print(f"\n--- Starting Episode (Items: {len(env.items_to_pack_this_episode)}, Containers: {env.num_containers}) ---")
    
    while not done:
        _, current_item_info = obs 
        if current_item_info is None: break 
        rand_c_idx = random.randint(0, env.num_containers - 1)
        sel_c_dims = env.container_dims_list[rand_c_idx] 
        rand_x = random.randint(0, sel_c_dims[0]-1) if sel_c_dims[0]>0 else 0
        rand_y = random.randint(0, sel_c_dims[1]-1) if sel_c_dims[1]>0 else 0
        rand_z = random.randint(0, sel_c_dims[2]-1) if sel_c_dims[2]>0 else 0
        rand_o_idx = random.randint(0, env.num_orientations - 1)
        action = (rand_c_idx, rand_x, rand_y, rand_z, rand_o_idx)
        item_name = env.items_to_pack_this_episode[env.current_item_idx_to_pack]['name']
        c_id_attempt = env.container_schemas[rand_c_idx]['id'] 
        obs, reward, done, info = env.step(action)
        total_reward+=reward; step_count+=1
        if step_count % 10 == 0 or info.get('packed_item_name') or info.get('skipped_item') or info.get('error'): 
            if info.get('packed_item_name'):
                c_idx_packed = info['packed_in_container_idx']; c_name_packed = env.container_schemas[c_idx_packed]['id']
                print(f"  S{step_count} OK: '{info['packed_item_name']}' in '{c_name_packed}'. Vol:{info['packed_volume_current_item']:.0f}. R:{reward:.1f}")
            elif info.get('skipped_item'): print(f"  S{step_count} SKIP: '{item_name}'. R:{reward:.1f}")
            elif info.get('error'): print(f"  S{step_count} ERR: '{item_name}' in '{c_id_attempt}' ({info['error']}). R:{reward:.1f}")
        if render_each_step and (info.get('packed_item_name') or info.get('skipped_item') or info.get('error')):
            env.render(fig=fig_render, title_prefix=f"Step {step_count}")
            plt.pause(0.02) 
        if done:
            print(f"--- Episode Finished ---"); print(f"Total Steps: {step_count}"); print(f"Total Reward: {total_reward:.2f}")
            print(f"Total Packed Volume (All Containers): {env.total_packed_volume:.2f}")
            for i in range(env.num_containers):
                c_id=env.container_schemas[i]['id']; c_vol=env.packed_volume_per_container[i]
                c_cap=np.prod(env.container_dims_list[i]); c_util=(c_vol/c_cap)*100 if c_cap>0 else 0
                print(f"  Container '{c_id}': PkdVol={c_vol:.0f}, Util={c_util:.1f}%")
            break
    if render_each_step: plt.ioff()
    if final_render:
        if fig_render is None: 
             fig_render = plt.figure(figsize=(sim_fig_width, sim_fig_height))
        env.render(fig=fig_render, title_prefix="Final Packing State")
        plt.show()
    elif render_each_step and fig_render: plt.show()
    return total_reward, env.total_packed_volume, env.packed_volume_per_container

if __name__ == '__main__':
    env = PackingEnv(
        container_schemas=TRAILER_SCHEMAS,
        item_definitions=ITEM_DEFINITIONS,
        item_sequence_names=ITEMS_TO_PACK_SEQUENCE
    )
    total_reward, final_total_volume, final_vol_per_container = run_random_agent_episode(
        env, render_each_step=False, final_render=True
    )
    print("\n--- Simulation Summary ---")
    print(f"Final Total Reward: {total_reward:.2f}")
    print(f"Final Total Packed Volume (All Containers): {final_total_volume:.2f}")
    for i in range(env.num_containers):
        c_id=env.container_schemas[i]['id']; c_vol=final_vol_per_container[i]
        c_cap=np.prod(env.container_dims_list[i]); c_util=(c_vol/c_cap)*100 if c_cap>0 else 0
        print(f"  Container '{c_id}': Final PkdVol={c_vol:.0f}, Final Util={c_util:.1f}%")

    print("\nDetails of Packed Items:")
    if env.packed_items_info:
        for item_info in env.packed_items_info:
            c_name = env.container_schemas[item_info['container_idx']]['id']
            print(f"  - {item_info['name']} in '{c_name}' at {item_info['pos']} "
                  f"dims {item_info['oriented_dims']}, Vol: {item_info['volume']:.0f}")
    else:
        print("  No items were successfully packed.")
