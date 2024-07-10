import copy
import random
import math
import numpy as np
from matplotlib import pyplot as plt
from deap import base, creator, tools

"""
    This file contains methods used in the genetic algorithm 
    and some other utils for plotting.

    Chromosome structure:
        [
            [0,1,2], # client_ids (n_clients=3)
            [0,0,1], # vehicles assigned to clients (n_vehicles=2)
            [4,5]    # vehicles max speed
            [3,3]    # vehicles max acceleration
        ]
"""

chridx_clients=0
chridx_vehicles=1
chridx_max_speeds=2
chridx_max_accs=3

def dist(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def angle(coord1, coord2):
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    return math.atan2(dy, dx)

def is_approaching(dist_from_dest):
    if len(dist_from_dest) > 2 and dist_from_dest[-1] > dist_from_dest[-2]:
        return False
    return True

def close_enough(coord_t, dest, dt):
    d = dist(coord_t, dest)
    return d <= 1*dt # do I reach dest with one step if I go with speed = 1 m/s?

class VehicleRoute():
    def __init__(self, chromo, exp_config, vehicle_id):
        """
            coords: list of coordinates of the clients in the route. Base coordinate at the beginning and end of the list.
        """
        self.vehicle_id = vehicle_id
        self.chromo = chromo
        self.exp_config = exp_config

        base_coord = self.exp_config["base_coord"]
        self.coords = [base_coord]
        n_clients = len(self.chromo[chridx_clients])
        for i in range(n_clients):
            veh = self.chromo[chridx_vehicles][i]
            if veh == self.vehicle_id:
                cli = self.chromo[chridx_clients][i]
                cli_coord = self.exp_config["clienst_coord"][cli]
                self.coords.append(cli_coord)
        self.coords.append(base_coord)

        self.dynamics = self.to_dynamics()

    def to_dynamics(self):
        """
            Finds the dynamics of the vehicle travelling the route (specified by self.coords).
            The dynamics include time, position (x,y), velocity, acceleration, battery soc, voltage, current, mechanical power, energy consumption over time.
            Time resolution: exp_config["dt"]
        """
        dynamics = {
            "time": [],
            "position": [],
            "velocity": [],
            "acceleration": [],
            "travelled_distance": [],
            "soc": [],
            "voltage": [],
            "current": [],
            "mech_power": [],
            "energy": [],
        }
        m = self.exp_config["vehicles_weight"][self.vehicle_id]
        b_capacity = self.exp_config["vehicles_battery_capacity"][self.vehicle_id]
        t = 0
        coord_t = self.exp_config["base_coord"]
        v_t = 0
        a_t = self.chromo[chridx_max_accs][self.vehicle_id]
        max_v = self.chromo[chridx_max_speeds][self.vehicle_id]
        soc_t = self.exp_config["vehicles_init_soc"][self.vehicle_id]

        soc_to_volt = lambda soc: (3.32 * math.exp(0.217 * soc)) - (0.784 * math.exp(-16.74 * soc))
        volt_t = soc_to_volt(soc_t)
        curr_t = 0.0
        mp_t = 0.0
        en_t = 0.0
        tra_dist_t = 0.0
        # print(self.coords)
        dt = self.exp_config["dt"]
        for i in range(len(self.coords)-1):
            source = self.coords[i].copy()
            dest = self.coords[i+1].copy()
            
            coord_t = source # in the previous step, coord_t has approached the prev. destination but didn't reach it exactly. Let's fix the error and set the coord_t value exactly to the client position
            dist_from_dest = [dist(coord_t, dest)]
            ang = angle(source, dest)
            v_t = 0.0

            # print(source, dest, is_approaching(dist_from_dest), close_enough(coord_t, dest, dt))
            while is_approaching(dist_from_dest) and not close_enough(coord_t, dest, dt):
                dynamics["time"].append(t)
                dynamics["position"].append(coord_t.copy())
                dynamics["velocity"].append(v_t)
                dynamics["acceleration"].append(a_t)
                dynamics["travelled_distance"].append(tra_dist_t)
                dynamics["soc"].append(soc_t)
                dynamics["voltage"].append(volt_t)
                dynamics["current"].append(curr_t)
                dynamics["mech_power"].append(mp_t)
                dynamics["energy"].append(en_t)


                # print(f"t: {t}, coord_t: {coord_t}, dest: {dest}, v_t: {v_t:.2f}, a_t: {a_t:.2f}, tra_dist_t: {tra_dist_t:.2f}, soc_t: {soc_t:.2f}, volt_t:{volt_t:.2f}, curr_t:{curr_t:.2f}, mp_t:{mp_t:.2f}, en_t:{en_t:.2f}")
                
                
                # start with a_t until v_t reaches max_v, then continue with a_t=0
                if v_t >= max_v:
                    a_t = 0.0 
                else:
                    a_t = self.chromo[chridx_max_accs][self.vehicle_id]
                t += dt
                v_t += a_t * dt

                d = v_t*dt
                tra_dist_t += d
                dx = d*math.cos(ang)
                dy = d*math.sin(ang)   
                coord_t[0] += dx
                coord_t[1] += dy
                # print(f"d: {d:.2f}, dx: {dx:.2f}, dy: {dy:.2f}, ang: {ang*180/3.14:.2f}, math.sin(ang): {math.sin(ang):.2f}, math.cos(ang): {math.cos(ang):.2f}")
                
                f = a_t*m + 0.5*1.32*0.109*1.2*v_t**2 + m*9.8*0.03
                mp_t = f*v_t
                
                curr_t = -mp_t / volt_t
                soc_t += (curr_t*dt/3600)/b_capacity
                soc_t = max(0.0, soc_t)
                # print(f"t: {t}, coord_t: {coord_t}, dest: {dest}, v_t: {v_t:.2f}, a_t: {a_t:.2f}, tra_dist_t: {tra_dist_t:.2f}, soc_t: {soc_t:.2f}, volt_t:{volt_t:.2f}, curr_t:{curr_t:.2f}, mp_t:{mp_t:.2f}, en_t:{en_t:.2f}")
                
                volt_t = soc_to_volt(soc_t)
                en_t += mp_t*dt

                # keep track of the distance from the destination (with the updated coord_t)
                dist_from_dest.append(dist(coord_t, dest))

                if soc_t == 0 or t > 1e6:
                    break

        # print(dynamics)
        for x in dynamics:
            dynamics[x] = np.array(dynamics[x])
        return dynamics

def create(exp_config):
    clients_list = np.arange(exp_config["num_clients"])
    random.shuffle(clients_list)
    schedule = [
            clients_list,  # Client ids in random order
            np.random.randint(exp_config["num_vehicles"], size=exp_config["num_clients"]), # vehicle id for each client
            exp_config["max_speed"]*np.random.random(size=(exp_config["num_vehicles"])),
            exp_config["max_acc"]*np.random.random(size=(exp_config["num_vehicles"]))
    ]
    return schedule

def eval(chromo, exp_config):
    routes = chromo_to_vehicle_routes(chromo, exp_config)
    total_energy = 0.0
    for ir, route in enumerate(routes):
        if len(route.dynamics["energy"]) > 0:
            total_energy += route.dynamics["energy"][-1]
    return [total_energy]

def crossover0(chromo1, chromo2):
    size = min(len(chromo1[chridx_clients]), len(chromo1[chridx_clients]))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    for x in [chridx_clients, chridx_vehicles]:
        chromo1[x][cxpoint1:cxpoint2], chromo2[x][cxpoint1:cxpoint2] \
            = chromo2[x][cxpoint1:cxpoint2], chromo1[x][cxpoint1:cxpoint2]
    
    tools.cxTwoPoint(chromo1[chridx_max_speeds], chromo2[chridx_max_speeds])
    tools.cxTwoPoint(chromo1[chridx_max_accs], chromo2[chridx_max_accs])

def crossover(chromo1, chromo2, exp_config):
    mer1, mer2 = eval(chromo1, exp_config), eval(chromo2, exp_config)
    if mer1 > mer2:
        return chromo1
    return chromo2

def mutate(chromo1):    
    tools.mutShuffleIndexes(chromo1[chridx_vehicles], indpb=0.5)[0]
    # print(chromo1[chridx_vehicles])
    tools.mutShuffleIndexes(chromo1[chridx_clients], indpb=0.5)[0]
    # print(chromo1[chridx_clients])
    tools.mutGaussian(chromo1[chridx_max_speeds], indpb=0.5, mu=5, sigma=3)[0]
    chromo1[chridx_max_speeds] = [max(1, x) for x in chromo1[chridx_max_speeds]]
    # print(chromo1[chridx_max_speeds])

    tools.mutGaussian(chromo1[chridx_max_accs], indpb=0.5, mu=5, sigma=3)[0]
    chromo1[chridx_max_accs] = [max(1, x) for x in chromo1[chridx_max_accs]]
    # print(chromo1[chridx_max_accs])

def plot_chromo(chromo, exp_config):
    print(exp_config)
    print(chromo)
    fig = plt.figure(figsize=(10,10))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    
    ## Plot the routes in x-y plane
    ax = subfigs[0].add_subplot(2,1,1)
    x = exp_config["base_coord"][0]
    y = exp_config["base_coord"][1]
    ax.scatter(x, y, color="k")
    ax.text(x, y, "base", color="k")

    ax.scatter(exp_config["clienst_coord"][:,0], exp_config["clienst_coord"][:,1], color="blue")
    for i in range(exp_config["num_clients"]):
        x = exp_config["clienst_coord"][i,0]
        y = exp_config["clienst_coord"][i,1]
        ax.text(x, y, f"client {i}", color="blue")
    
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    NUM_COLORS = exp_config["num_vehicles"]
    cm = plt.get_cmap('viridis')
    colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    routes = chromo_to_vehicle_routes(chromo, exp_config)
    for ir, route in enumerate(routes):
        for ic in range(len(route.coords)-1):
            x,y = route.coords[ic]
            x2, y2 = route.coords[ic+1]
            # ax.arrow(x,y, (x2-x), (y2-y), color=colors[ir], head_width=15, length_includes_head=True)
            ax.annotate("", xy=(x, y), xytext=(x2, y2),
                arrowprops=dict(arrowstyle="<-", color=colors[ir]))
        print(route.dynamics["position"].shape)
        if len(route.dynamics["position"] > 0):
            x = route.dynamics["position"][:,0]
            y = route.dynamics["position"][:,1]
            ax.scatter(x, y, color="pink", s=4)

    # Plot vehicle dyanmics in the route
    axes = subfigs[1].subplots(7, 1, sharex=True)
    for ir, route in enumerate(routes):
        t = route.dynamics["time"]

        i = 0
        ax = axes[i]
        ax.plot(t, route.dynamics["velocity"], color=colors[ir])
        ax.set_ylabel("Speed (m/s)")
        i+=1

        ax = axes[i]
        ax.plot(t, route.dynamics["acceleration"], color=colors[ir])
        ax.set_ylabel("Acc. (m/s2)")
        i+=1

        ax = axes[i]
        ax.plot(t, route.dynamics["soc"], color=colors[ir])
        ax.set_ylabel("SoC")
        i+=1

        ax = axes[i]
        ax.plot(t, route.dynamics["current"], color=colors[ir])
        ax.set_ylabel("Current (A)")
        i+=1

        ax = axes[i]
        ax.plot(t, route.dynamics["voltage"], color=colors[ir])
        ax.set_ylabel("Voltage (V)")
        i+=1

        ax = axes[i]
        ax.plot(t, route.dynamics["mech_power"], color=colors[ir])
        ax.set_ylabel("Power (W)")
        i+=1

        ax = axes[i]
        ax.plot(t, route.dynamics["energy"], color=colors[ir])
        ax.set_ylabel("Energy (J)")
        i+=1
        
    plt.show()

def chromo_to_vehicle_routes(chromo, exp_config):
    routes = []
    for vid in range(exp_config["num_vehicles"]):
        route = VehicleRoute(chromo, exp_config, vid)
        routes.append(route)
    return routes        

def run(exp_config, num_generations=5):
    tb = base.Toolbox()

    creator.create('Fitness_Func', base.Fitness, weights=(-1,))
    creator.create('Individual', list, fitness=creator.Fitness_Func)

    tb.register('indexes', create, exp_config=exp_config)
    tb.register('individual', tools.initIterate, creator.Individual, tb.indexes)
    tb.register('population', tools.initRepeat, list, tb.individual)
    tb.register('evaluate', eval, exp_config=exp_config)
    tb.register('select', tools.selTournament)
    tb.register('mutate', mutate)
    tb.register("mate", crossover, exp_config=exp_config)
    
    prob_crossover = 0.8
    prob_mutation = 0.5
    num_population = 1000

    population = tb.population(n=num_population)

    fitness_set = list(tb.map(tb.evaluate, population))
    for ind, fit in zip(population, fitness_set):
        ind.fitness.values = fit

    best_fit_list = []
    best_sol_list = []

    best_fit = float('inf')

    for gen in range(num_generations):
        print(f'Generation: {gen:4} | Fitness: {best_fit:.2f}')

        offspring = tb.select(population, len(population), tournsize=3)
        offspring = list(map(tb.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < prob_crossover:
                tb.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for chromo in offspring:
            if np.random.random() < prob_mutation:
                tb.mutate(chromo)
                del chromo.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness_set = map(tb.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness_set):
            ind.fitness.values = fit

        population[:] = offspring

        curr_best_sol = tools.selBest(population, 1)[0]
        
        plot_chromo(curr_best_sol, exp_config)

        curr_best_fit = curr_best_sol.fitness.values[0]

        if curr_best_fit < best_fit:
            best_sol = curr_best_sol
            best_fit = curr_best_fit

        best_fit_list.append(best_fit)
        best_sol_list.append(best_sol)

    return best_fit_list, best_sol