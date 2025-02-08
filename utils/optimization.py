import numpy as np

def fitness_function(individual, lines, points, threshold):
    import math
    import numpy as np

    delta_x, delta_y = individual
    points1 = points.copy()
    points1[:,0] = points[:,0] + delta_x
    points1[:,1] = points[:,1] + delta_y
    # print(f'points={points}, points1={points1}')

    count_within_threshold = 0
    distances = []
    linear_distance = []

    for (point1, line) in zip(points1, lines):
        x, y = point1
        a, b, c = line
        distance = (abs(a * x + b * y + c) / math.sqrt((a ** 2 + b ** 2)))
        distances.append(distance)
        linear_distance.append(abs(b * delta_x - a * delta_y) / math.sqrt(a ** 2 + b ** 2))
        if distance < threshold:
            count_within_threshold += 1
    # print(f'distances={distances},mean={np.mean(distances)}')

    middle_mean = np.mean(distances) 

    return middle_mean - 0.5*count_within_threshold + 0.1 * np.mean(linear_distance)


def differential_evolution(fitness_function, num_generations, lines, points, threshold):
    opt_lr = 5
    population = [[0, 0]]
    new_population = []
    for generation in range(num_generations):
        if generation > 0:
            opt_lr = 5
        if generation > 10:
            opt_lr = 3
        if generation > 20:
            opt_lr = 1
        if generation > 30:
            opt_lr = 0.5
        if generation > 40:
            opt_lr = 0.1

        for i in population:
            for ii in [-2,-1,0,1,2]:
                for jj in [-2,-1,0,1,2]:
                    if [i[0] + ii*opt_lr, i[1] + jj*opt_lr] not in new_population:
                        # print(f'i={i},ii={ii},jj={jj}')
                        new_population.append([i[0] + ii*opt_lr, i[1] + jj*opt_lr])
        score_list = [fitness_function(x, lines, points, threshold) for x in new_population]
        # print(f'new_population={new_population}')
        population = np.array(new_population)[np.array(score_list).argsort()[:15]]
        # print(f'generation={generation},population={population},score_list={score_list}')
    population = np.array(new_population)
    best_individual = population[np.argmin([fitness_function(x, lines, points, threshold) for x in population])]
    # print(f'min={np.argmin([fitness_function(x, lines, points, threshold) for x in population])}')
    # print(f'best_individual={best_individual}')
    return best_individual
