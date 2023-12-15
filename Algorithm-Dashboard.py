import streamlit as st
import math
import heapq
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import permutations
from collections import Counter
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


st.set_page_config(page_title="Algorithm Toolbox", page_icon="🧮")
st.title("🔍 Algorithm Dashboard")

# with open(r'C:\Users\tariq.aziz\OneDrive - University of Central Asia\Desktop\style.css') as f:
#     st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def bubble_sort(arr):
    n = len(arr)
    iterations = []
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                iterations.append(arr.copy())
    return arr, iterations


def selection_sort(arr):
    n = len(arr)
    iterations = []
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        iterations.append(arr.copy())
    return arr, iterations


def insertion_sort(arr):
    n = len(arr)
    iterations = []
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        iterations.append(arr.copy())
    return arr, iterations


def merge_sort(arr):
    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def _merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        left = _merge_sort(left)
        right = _merge_sort(right)
        return merge(left, right)
    n = len(arr)
    iterations = []
    for step in range(1, n + 1):
        current_iteration = _merge_sort(arr[:step])
        iterations.append(current_iteration)
    return iterations


def binary_search_visualization(arr, target):
    arr.sort()
    n = len(arr)
    low = 0
    high = n - 1
    steps = []
    while low <= high:
        mid = (low + high) // 2
        current_step = {
            'low': low,
            'high': high,
            'mid': mid,
            'value_mid': arr[mid],
        }
        steps.append(current_step)
        if arr[mid] < target:
            low = mid + 1
        elif arr[mid] > target:
            high = mid - 1
        else:
            return steps, mid
    return steps, -1


def master_theorem(a, b, c):
    if a < 1 or b <= 1:
        return "a must be greater than 0 and b must be greater than 1."
    log_b_a = math.log(a, b)
    if a > b**c:
        return f"T(n) = Θ(n^{log_b_a})"
    elif a == b**c:
        return f"T(n) = Θ(n^{c} * log(n))"
    else:
        return f"T(n) = Θ(n^{c})"


# Function to solve the coin change problem for the total number of ways
def coin_change_ways(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    return dp[amount]


def min_coins_and_combination(coins, amount):
    dp = [float('inf')] * (amount + 1)
    coin_used = [0] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] + 1 < dp[x]:
                dp[x] = dp[x - coin] + 1
                coin_used[x] = coin
    if dp[amount] == float('inf'):
        return -1, []
    coins_combination = []
    while amount > 0:
        coins_combination.append(coin_used[amount])
        amount -= coin_used[amount]
    return dp[-1], coins_combination


def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1]
                               [w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    w = capacity
    items_included = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items_included.append(i-1)
            w -= weights[i-1]
    return dp[n][capacity], items_included


def fractional_knapsack(values, weights, capacity):
    ratio_and_items = sorted(
        [[v/w, w, i] for i, (v, w) in enumerate(zip(values, weights))], reverse=True)
    total_value = 0
    item_amounts = [0] * len(values)
    for ratio, weight, index in ratio_and_items:
        if capacity >= weight:
            capacity -= weight
            total_value += weight * ratio
            item_amounts[index] = weight
        else:
            total_value += capacity * ratio
            item_amounts[index] = capacity
            break

    return total_value, item_amounts


# Define the knapsack problem parameters
knapsack_capacity = 12
items = [
    {"weight": 5, "value": 12},
    {"weight": 3, "value": 5},
    {"weight": 7, "value": 10},
    {"weight": 2, "value": 7}
]


def huffman_coding(data):
    frequency = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    codes = {symbol: code for weight, *pairs in heap for symbol, code in pairs}
    return codes


def huffman_encoding(data):
    codes = huffman_coding(data)
    encoded_data = ''.join([codes[symbol] for symbol in data])
    return encoded_data, codes


def huffman_decoding(encoded_data, codes):
    reverse_codes = {code: symbol for symbol, code in codes.items()}
    decoded_data = ""
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            symbol = reverse_codes[current_code]
            decoded_data += symbol
            current_code = ""
    return decoded_data


# Through dynamic programming
def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def branch_and_bound(job_matrix):
    n = len(job_matrix)
    assigned_jobs = np.zeros(n, dtype=int)
    min_cost = [float("inf")]

    def branch_and_bound_util(assigned_jobs, cost_so_far, level):
        nonlocal min_cost
        if level == n:
            min_cost[0] = min(min_cost[0], cost_so_far)
            return
        for i in range(n):
            if not assigned_jobs[i]:
                new_assignment = assigned_jobs.copy()
                new_assignment[i] = 1
                if cost_so_far + cost(level, i) < min_cost[0]:
                    branch_and_bound_util(
                        new_assignment, cost_so_far + cost(level, i), level + 1
                    )

    def cost(row, col):
        return job_matrix[row][col]
    branch_and_bound_util(assigned_jobs, 0, 0)
    return min_cost[0]


vertices = ['A', 'B', 'C', 'D', 'E']
graph = {
    'A': {'B', 'D', 'E'},
    'B': {'A', 'C', 'D', 'E'},
    'C': {'B', 'D', 'E'},
    'D': {'A', 'B', 'C', 'E'},
    'E': {'A', 'B', 'C', 'D'}
}


def minimum_vertex_cover(graph):
    vertex_cover = set()
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    while graph.edges():
        edge = random.choice(list(graph.edges()))
        vertex_cover.add(edge[0])
        vertex_cover.add(edge[1])
        graph.remove_node(edge[0])
        graph.remove_node(edge[1])
    return vertex_cover


# Genetic Algorithm Functions
def generate_population(population_size, target_length):
    return [''.join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(target_length)) for _ in range(population_size)]


def calculate_fitness(individual, target):
    return sum(1 for i, j in zip(individual, target) if i == j)


def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.choices(population, k=2)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    return random.choices(population, weights=selection_probs, k=2)


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate=0.1):
    mutated_individual = [c if random.random() > mutation_rate else random.choice(
        "abcdefghijklmnopqrstuvwxyz ") for c in individual]
    return ''.join(mutated_individual)


def run_genetic_algorithm(target, generations=100, population_size=20, mutation_rate=0.1):
    population = generate_population(population_size, len(target))
    fitness_scores = [calculate_fitness(
        individual, target) for individual in population]
    best_fitness = max(fitness_scores)
    best_individual = population[fitness_scores.index(best_fitness)]
    fitness_history = [best_fitness]
    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend(
                [mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        population = new_population
        fitness_scores = [calculate_fitness(
            individual, target) for individual in population]
        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_scores.index(best_fitness)]
        fitness_history.append(best_fitness)
    return best_individual, fitness_history


class Attraction:
    def __init__(self, name, rating, cost, travel_time):
        self.name = name
        self.rating = rating
        self.cost = cost
        self.travel_time = travel_time


attractions = [
    Attraction("Museum of Art", 4.5, 2, 5),
    Attraction("Historical Park", 4.0, 5, 1),
    Attraction("Science Center", 4.2, 5, 2),
    Attraction("Botanical Garden", 3.8, 1, 5),
    Attraction("City Zoo", 4.1, 2, 5),
    Attraction("Local Zoo", 4.1, 2, 5),
]


def tsp_dynamic_programming(attractions):
    n = len(attractions)
    all_points = range(n)
    memo = {}

    def distance(point1, point2):
        return ((point1.cost - point2.cost)**2 + (point1.travel_time - point2.travel_time)**2)**0.5

    def tsp_memo(mask, last_point):
        if mask == (1 << n) - 1:
            return distance(attractions[last_point], attractions[0]), [0]
        if (mask, last_point) in memo:
            return memo[(mask, last_point)]
        min_distance = float('inf')
        optimal_order = None
        for next_point in all_points:
            if (mask >> next_point) & 1 == 0:
                new_distance, new_order = tsp_memo(
                    mask | (1 << next_point), next_point)
                new_distance += distance(attractions[last_point],
                                         attractions[next_point])
                if new_distance < min_distance:
                    min_distance = new_distance
                    optimal_order = [last_point] + new_order
        memo[(mask, last_point)] = min_distance, optimal_order
        return min_distance, optimal_order
    total_distance, optimal_order = tsp_memo(1, 0)
    return total_distance, [attractions[i] for i in optimal_order]


# Function for the Greedy Approach (Activity Selection Problem)
def greedy_activity_selection(start_times, end_times):
    activities = list(zip(start_times, end_times))
    activities.sort(key=lambda x: x[1])
    selected_activities = [activities[0]]

    for activity in activities[1:]:
        if activity[0] >= selected_activities[-1][1]:
            selected_activities.append(activity)
    return selected_activities


# Function to find the maximum clique using backtracking
def find_max_clique(graph, clique, candidates, result):
    if not candidates and len(clique) > len(result):
        result.clear()
        result.extend(clique)
    while candidates:
        node = candidates.pop()
        new_clique = [n for n in clique if n in graph[node]]
        new_candidates = [n for n in candidates if n in graph[node]]
        find_max_clique(graph, new_clique + [node], new_candidates, result)


def visualize_max_clique(graph, max_clique):
    g = nx.Graph(graph)
    plt.figure()
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos)
    plt.title("Original Graph")
    plt.axis("off")
    st.pyplot(plt)
    g_clique = g.subgraph(max_clique)
    plt.figure()
    pos_clique = nx.spring_layout(g_clique)
    nx.draw_networkx_nodes(g_clique, pos_clique)
    nx.draw_networkx_edges(g_clique, pos_clique)
    nx.draw_networkx_labels(g_clique, pos_clique)
    plt.title("Maximum Clique")
    plt.axis("off")
    st.pyplot(plt)


graph = {
    'A': {'B', 'C', 'D'},
    'B': {'A', 'C'},
    'C': {'A', 'B', 'D'},
    'D': {'A', 'C'},
}


# Function to find the maximum independent set using backtracking
def find_max_independent_set(graph, independent_set, candidates, result):
    if not candidates and len(independent_set) > len(result):
        result.clear()
        result.extend(independent_set)
    while candidates:
        node = candidates.pop()
        new_independent_set = [
            n for n in independent_set if n not in graph[node]]
        new_candidates = [n for n in candidates if n not in graph[node]]
        find_max_independent_set(
            graph, new_independent_set + [node], new_candidates, result)


def visualize_max_independent_set(graph, max_independent_set):
    g = nx.Graph(graph)
    plt.figure()
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos)
    plt.title("Original Graph")
    plt.axis("off")
    st.pyplot(plt)
    g_independent_set = g.subgraph(max_independent_set)
    plt.figure()
    pos_independent_set = nx.spring_layout(g_independent_set)
    nx.draw_networkx_nodes(g_independent_set, pos_independent_set)
    nx.draw_networkx_edges(g_independent_set, pos_independent_set)
    nx.draw_networkx_labels(g_independent_set, pos_independent_set)
    plt.title("Maximum Independent Set")
    plt.axis("off")
    st.pyplot(plt)


# Function to check if the formula is satisfied with the given assignment
def is_satisfied(formula, assignment):
    for clause in formula:
        clause_satisfied = False
        for literal in clause:
            variable, is_negated = literal[0], literal[1]
            value = not assignment[variable] if is_negated else assignment[variable]
            clause_satisfied |= value
        if not clause_satisfied:
            return False
    return True


def backtracking_satisfy(formula, assignment, variable_index):
    if variable_index == len(assignment):
        return is_satisfied(formula, assignment)
    current_variable = list(assignment.keys())[variable_index]
    assignment[current_variable] = False
    if backtracking_satisfy(formula, assignment, variable_index + 1):
        return True
    assignment[current_variable] = True
    if backtracking_satisfy(formula, assignment, variable_index + 1):
        return True
    return False


def hamiltonian_path(graph):
    for perm in permutations(graph.nodes()):
        if all(graph.has_edge(perm[i], perm[i+1]) for i in range(len(perm)-1)):
            return list(perm)
    return None


def hamiltonian_cycle(graph):
    for perm in permutations(graph.nodes()):
        if all(graph.has_edge(perm[i], perm[i+1]) for i in range(len(perm)-1)):
            return list(perm) + [perm[0]]
    return None


def minimum_spanning_tree(graph):
    mst_edges = set()
    vertices = list(graph.keys())
    start_vertex = vertices[0]
    visited = set([start_vertex])
    edges = [
        (cost, start_vertex, to)
        for to, cost in graph[start_vertex].items()
    ]
    heapq.heapify(edges)
    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst_edges.add((frm, to, cost))
            for to_next, cost_next in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (cost_next, to, to_next))
    return mst_edges


vertices = ['A', 'B', 'C', 'D', 'E']
graph = {
    'A': {'B': 2, 'D': 5, 'E': 1},
    'B': {'A': 2, 'C': 1, 'D': 3, 'E': 2},
    'C': {'B': 1, 'D': 1, 'E': 4},
    'D': {'A': 5, 'B': 3, 'C': 1, 'E': 3},
    'E': {'A': 1, 'B': 2, 'C': 4, 'D': 3}
}


def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0

    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances


def kruskal(graph):
    edges = [
        (cost, frm, to)
        for frm, edges in graph.items()
        for to, cost in edges.items()
    ]
    edges.sort()
    mst_edges = set()
    disjoint_sets = {vertex: {vertex} for vertex in graph}
    for cost, frm, to in edges:
        if disjoint_sets[frm] != disjoint_sets[to]:
            mst_edges.add((frm, to, cost))
            union_set = disjoint_sets[frm].union(disjoint_sets[to])
            for vertex in union_set:
                disjoint_sets[vertex] = union_set

    return mst_edges


def prim(graph, start):
    mst_edges = set()
    visited = {vertex: False for vertex in graph}
    visited[start] = True
    priority_queue = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(priority_queue)

    while priority_queue:
        cost, frm, to = heapq.heappop(priority_queue)
        if visited[to]:
            continue
        visited[to] = True
        mst_edges.add((frm, to, cost))

        for neighbor, weight in graph[to].items():
            if not visited[neighbor]:
                heapq.heappush(priority_queue, (weight, to, neighbor))
    return mst_edges


# Category Selection
st.subheader("Select Algorithm Category")
algorithm_category = st.selectbox(
    "",
    ("Sorting Algorithm", "Search Algorithm", "Master Theorem", "Coin Change", "Knapsack", "Fractional Knapsack", "Dynamic Programming", "Branch and Bound", "Genetic Algorithm", "Travelling Salesman Problem", "Greedy Algorithm",
     "Minimum Spanning Tree", "Dijkstra's Algorithm", "Kruskal's Algorithm", "Prim's Algorithm", "Minimum Vertex Cover", "Clique Problem", "Maximum Independent Set Problem", "Boolean Satisfiability Problem", "Hamiltonian Cycle", "Hamiltonian Path")
)

st.markdown("---")


# Different Sections for Each Category
if algorithm_category == "Sorting Algorithm":
    st.subheader("🔀 Sorting Algorithms")
    algorithm = st.selectbox(
        "Choose a Sorting Algorithm",
        ("Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort")
    )

    time_complexities = {
        "Bubble Sort": "O(n^2)",
        "Selection Sort": "O(n^2)",
        "Insertion Sort": "O(n^2)",
        "Merge Sort": "O(n log n)",
        "Binary Search": "O(log n)"
    }
    st.write(f"Time Complexity: {time_complexities[algorithm]}")

    arr_input = st.text_input("Enter a list of numbers, separated by commas")
    arr = [int(x) for x in arr_input.split(",")] if arr_input else []


elif algorithm_category == "Search Algorithm":
    st.subheader("🔍 Search Algorithms")
    algorithm = st.selectbox("Choose a Search Algorithm", ("Binary Search",))
    arr_input = st.text_area(
        "Enter a sorted list of numbers, separated by commas")
    arr = [int(x) for x in arr_input.split(",")] if arr_input else []
    target = st.number_input("Enter a target value", step=1)


if st.button("Execute Algorithm"):
    result = None
    iterations = None
    if algorithm == "Bubble Sort":
        result, iterations = bubble_sort(arr.copy())
        if iterations:
            fig = go.Figure()
            for i, iteration in enumerate(iterations):
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(iteration))),
                        y=iteration,
                        name=f"Step {i + 1}",
                        marker_color='skyblue',
                        showlegend=(i == 0),
                    )
                )
            fig.update_layout(
                title="Bubble Sort Visualization",
                xaxis_title="Index",
                yaxis_title="Value",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    }]
                }],
            )
            frames = [go.Frame(data=[go.Bar(x=list(range(len(iteration))), y=iteration, marker_color='skyblue')], name=f"Step {i + 1}")
                      for i, iteration in enumerate(iterations)]
            fig.frames = frames
            st.plotly_chart(fig)

    elif algorithm == "Selection Sort":
        result, iterations = selection_sort(arr.copy())
        if iterations:
            fig = go.Figure()
            for i, iteration in enumerate(iterations):
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(iteration))),
                        y=iteration,
                        name=f"Step {i + 1}",
                        marker_color='skyblue',
                        showlegend=(i == 0),
                    )
                )
            fig.update_layout(
                title=f"{algorithm} Visualization",
                xaxis_title="Index",
                yaxis_title="Value",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    }]
                }],
            )
            frames = [go.Frame(data=[go.Bar(x=list(range(len(iteration))), y=iteration, marker_color='skyblue')], name=f"Step {i + 1}")
                      for i, iteration in enumerate(iterations)]
            fig.frames = frames
            st.plotly_chart(fig)

    elif algorithm == "Insertion Sort":
        result, iterations = insertion_sort(arr.copy())
        if iterations:
            fig = go.Figure()
            for i, iteration in enumerate(iterations):
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(iteration))),
                        y=iteration,
                        name=f"Step {i + 1}",
                        marker_color='skyblue',
                        showlegend=(i == 0),
                    )
                )
            fig.update_layout(
                title=f"{algorithm} Visualization",
                xaxis_title="Index",
                yaxis_title="Value",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    }]
                }],
            )
            frames = [go.Frame(data=[go.Bar(x=list(range(len(iteration))), y=iteration, marker_color='skyblue')], name=f"Step {i + 1}")
                      for i, iteration in enumerate(iterations)]
            fig.frames = frames
            st.plotly_chart(fig)

    elif algorithm == "Merge Sort":
        iterations = merge_sort(arr.copy())
        st.write("Original List:", arr)
        st.write("Sorted List:", iterations[-1])
        if iterations:
            fig = go.Figure()
            for i, iteration in enumerate(iterations):
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(iteration))),
                        y=iteration,
                        name=f"Step {i + 1}",
                        marker_color='skyblue',
                        showlegend=(i == 0),
                    )
                )

            fig.update_layout(
                title=f"{algorithm} Visualization",
                xaxis_title="Index",
                yaxis_title="Value",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    }]
                }],
            )
            frames = [go.Frame(data=[go.Bar(x=list(range(len(iteration))), y=iteration, marker_color='skyblue')], name=f"Step {i + 1}")
                      for i, iteration in enumerate(iterations)]
            fig.frames = frames
            st.plotly_chart(fig)

    elif algorithm == "Binary Search":
        steps, result = binary_search_visualization(arr.copy(), target)
        st.write("Original List:", arr)
        arr.sort()
        st.write("Sorted List:", arr)
        st.write("Target Value:", target)
        if result != -1:
            st.write(f"Element found at index: {result}")
        else:
            st.write("Element not found")
        if steps:
            fig = go.Figure()

            for i, step in enumerate(steps):
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(arr))),
                        y=arr,
                        marker_color='skyblue',
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[step['mid']],
                        y=[step['value_mid']],
                        mode='markers',
                        marker=dict(color='red', size=10),
                        text=f'Step {i + 1}<br>Mid: {step["mid"]}<br>Value: {step["value_mid"]}',
                    )
                )
                fig.add_shape(
                    go.layout.Shape(
                        type='line',
                        x0=step['mid'],
                        y0=min(arr),
                        x1=step['mid'],
                        y1=max(arr),
                        line=dict(color='red', width=2),
                    )
                )
            fig.update_layout(
                title="Binary Search Visualization",
                xaxis_title="Index",
                yaxis_title="Value",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                    }]
                }],
            )
            st.plotly_chart(fig)
    st.write("Result:", result)


elif algorithm_category == "Master Theorem":
    st.subheader("📚 Master Theorem")
    st.markdown("""
    ### Master Theorem Cases
    - **Case 1**: If \( a > b^c \) then \( T(n) = O(n^{log a}) \)
    - **Case 2**: If \( a = b^c \) then \( T(n) = O(n^c * log n) \)
    - **Case 3**: If \( a < b^c \) then \( T(n) = O(n^c) \)
    """)
    a_mt = st.number_input(
        "Enter a for Master Theorem", min_value=1, value=2)
    b_mt = st.number_input(
        "Enter b for Master Theorem", min_value=2, value=2)
    c_mt = st.number_input(
        "Enter c for Master Theorem", value=0)

    if st.button("Calculate using Master Theorem"):
        result_mt = master_theorem(a_mt, b_mt, c_mt)
        st.write(result_mt)


elif algorithm_category == "Coin Change":
    st.subheader("💰 Coin Change")
    coins_input = st.text_input(
        "Enter the coin denominations (comma-separated)", "1, 2, 5")
    amount = st.number_input("Enter the amount", min_value=0, value=0)
    if st.button("Calculate"):
        coins_list = list(map(int, coins_input.split(',')))
        total_ways = coin_change_ways(coins_list, amount)
        minimum_coins, coins_combination = min_coins_and_combination(
            coins_list, amount)
        st.write(f"Number of ways to make change for {amount}: {total_ways}")
        if minimum_coins != -1:
            st.write(
                f"Minimum number of coins needed for {amount}: {minimum_coins}")
            st.write("Coins used:", coins_combination)
            dp_table = [[0] * (amount + 1) for _ in range(len(coins_list) + 1)]
            for i in range(len(coins_list) + 1):
                dp_table[i][0] = 1
            for i in range(1, len(coins_list) + 1):
                for j in range(1, amount + 1):
                    dp_table[i][j] = dp_table[i - 1][j]
                    if j >= coins_list[i - 1]:
                        dp_table[i][j] += dp_table[i][j - coins_list[i - 1]]
            st.subheader("Dynamic Programming Table Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(dp_table, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_xlabel("Amount")
            ax.set_ylabel("Coin Index")
            ax.set_title("Dynamic Programming Table")
            st.pyplot(fig)

        else:
            st.write("No solution")


elif algorithm_category == "Knapsack":
    st.subheader("🎒 Knapsack Problem")
    values_input = st.text_input(
        "Enter the values (comma-separated)", "60, 100, 120")
    weights_input = st.text_input(
        "Enter the weights (comma-separated)", "10, 20, 30")
    capacity = st.number_input(
        "Enter the knapsack capacity", min_value=0, value=0)
    if st.button("Calculate"):
        values = list(map(int, values_input.split(',')))
        weights = list(map(int, weights_input.split(',')))
        max_value, items_included = knapsack(values, weights, capacity)
        st.write(f"Maximum value in the knapsack: {max_value}")
        st.write("Items included (0-based index):", items_included)
        st.subheader("Knapsack Items Visualization")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(values)), values,
                color='lightblue', label='Item Value')
        plt.bar(items_included, [
                values[i] for i in items_included], color='blue', label='Included in Knapsack')
        plt.xlabel("Item Index")
        plt.ylabel("Value")
        plt.title("Knapsack Items and Inclusion")
        plt.legend()
        st.pyplot(plt)


elif algorithm_category == "Fractional Knapsack":
    st.subheader("🎒 Fractional Knapsack Problem")
    values_input = st.text_input(
        "Enter the values (comma-separated)", "12, 5, 10, 7")
    weights_input = st.text_input(
        "Enter the weights (comma-separated)", "5, 3, 7, 2")
    capacity = st.number_input(
        "Enter the knapsack capacity", min_value=0, value=0)
    if st.button("Calculate"):
        values = list(map(int, values_input.split(',')))
        weights = list(map(int, weights_input.split(',')))
        total_value, item_amounts = fractional_knapsack(
            values, weights, capacity)
        st.write(f"Total value in the knapsack: {total_value}")
        st.write("Item amounts:", item_amounts)
        st.subheader("Fractional Knapsack Visualization")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(values)), values,
                color='lightblue', label='Item Value')
        plt.bar(range(len(weights)), weights,
                color='lightgreen', label='Item Weight')
        plt.bar(range(len(item_amounts)), item_amounts,
                color='blue', alpha=0.5, label='Included Amount')
        plt.xlabel("Item Index")
        plt.ylabel("Value/Weight")
        plt.title("Fractional Knapsack Items and Inclusion")
        plt.legend()
        st.pyplot(plt)


elif algorithm_category == "Huffman Coding":
    st.subheader("🔠 Huffman Coding")
    input_text = st.text_area("Enter the text to be encoded", "")
    if st.button("Encode Text"):
        if not input_text:
            st.warning("Please enter text for encoding.")
        else:
            encoded_data, codes_huffman = huffman_encoding(input_text)
            st.write("Original Text:", input_text)
            st.write("Encoded Data:", encoded_data)
            st.write("Codes:", codes_huffman)
    encoded_data = st.text_input("Enter the encoded data for decoding", "")
    if st.button("Decode Text (Huffman)"):
        if 'codes_huffman' in locals():
            decoded_data_huffman = huffman_decoding(
                encoded_data, codes_huffman)
            st.write("Decoded Data (Huffman):", decoded_data_huffman)
        else:
            st.warning("Please encode the text using Huffman coding first.")


elif algorithm_category == "Dynamic Programming":
    st.subheader("🔍 Dynamic Programming - Longest Common Subsequence (LCS)")
    st.markdown("""
    The Longest Common Subsequence (LCS) problem is to find the length of the longest subsequence 
    common to two given sequences.
    """)
    str1 = st.text_input("Enter the first string:")
    str2 = st.text_input("Enter the second string:")
    if st.button("Find LCS Length"):
        lcs_length = longest_common_subsequence(str1, str2)
        st.write(f"Length of Longest Common Subsequence (LCS): {lcs_length}")


elif algorithm_category == "Branch and Bound":
    st.subheader("🌐 Branch and Bound - Job Assignment Problem")
    st.markdown("""
    The Job Assignment Problem is to assign n jobs to n workers, such that the total cost is minimized.
    """)
    n_jobs = st.number_input(
        "Enter the number of jobs/workers:", min_value=1, value=3)
    job_matrix = []
    st.write("Enter the cost matrix for the Job Assignment Problem:")
    for i in range(n_jobs):
        row = st.text_input(f"Enter costs for job {i+1} (comma-separated):")
        if row:
            job_matrix.append(list(map(int, row.split(','))))
        else:
            st.error("Please enter valid costs for the job.")
    job_matrix = np.array(job_matrix)
    if st.button("Find Minimum Cost (Branch and Bound)"):
        min_cost = branch_and_bound(job_matrix)
        st.write(f"Minimum Cost for Job Assignment: {min_cost}")
        fig = px.imshow(job_matrix, labels=dict(
            x="Worker", y="Job"), title="Job Assignment Matrix")
        st.plotly_chart(fig)


elif algorithm_category == "Genetic Algorithm":
    st.subheader("🧬 Genetic Algorithm - String Evolution")
    target_string = st.text_input("Enter the target string:", "hello world")
    generations = st.number_input(
        "Enter the number of generations:", min_value=1, value=100)
    population_size = st.number_input(
        "Enter the population size:", min_value=2, value=20)
    mutation_rate = st.slider(
        "Select the mutation rate:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    if st.button("Run Genetic Algorithm"):
        best_individual, fitness_history = run_genetic_algorithm(
            target_string, generations, population_size, mutation_rate)
        st.write(f"Best Individual: {best_individual}")
        st.write(
            f"Best Fitness: {calculate_fitness(best_individual, target_string)}")
        plt.plot(np.arange(generations + 1), fitness_history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Genetic Algorithm Fitness Over Generations")
        st.pyplot(plt)


elif algorithm_category == "Travelling Salesman Problem":
    st.subheader("🌍 Travelling Salesman Problem")
    if attractions:
        fig_original, ax_original = plt.subplots()
        for i, attraction in enumerate(attractions):
            ax_original.scatter(
                attraction.cost, attraction.travel_time, label=attraction.name)
            ax_original.text(attraction.cost, attraction.travel_time,
                             f"{attraction.name}\nRating: {attraction.rating}\nCost: {attraction.cost}\nTime: {attraction.travel_time}", fontsize=8)
        ax_original.set_xlabel('Cost')
        ax_original.set_ylabel('Travel Time')
        ax_original.set_title('Original Attractions')
        st.pyplot(fig_original)
        if st.button("Find Optimal Tour"):
            total_distance, optimal_order = tsp_dynamic_programming(
                attractions)
            fig_optimal, ax_optimal = plt.subplots()
            for i, attraction in enumerate(attractions):
                color = 'blue' if i != 0 else 'red'
                ax_optimal.scatter(
                    attraction.cost, attraction.travel_time, label=attraction.name, color=color)
                ax_optimal.text(attraction.cost,
                                attraction.travel_time, str(i), fontsize=12)
            for i in range(len(optimal_order) - 1):
                start_point = optimal_order[i]
                end_point = optimal_order[i + 1]
                ax_optimal.plot([start_point.cost, end_point.cost], [
                                start_point.travel_time, end_point.travel_time], 'k-')
            ax_optimal.set_xlabel('Cost')
            ax_optimal.set_ylabel('Travel Time')
            ax_optimal.set_title('Optimal Tour')
            st.write(f"Optimal Tour Distance: {total_distance:.2f}")
            st.pyplot(fig_optimal)
    else:
        st.warning("No attractions available. Please add attractions first.")


elif algorithm_category == "Greedy Algorithm":
    st.subheader("🌐 Greedy Algorithm - Activity Selection Problem")
    st.markdown("""
    The Activity Selection Problem is a classic example of a greedy algorithm. 
    Given a set of activities with start and end times, the goal is to select 
    the maximum number of non-overlapping activities.
    """)
    start_times_input = st.text_input(
        "Enter the start times of activities (comma-separated)", "1, 3, 0, 5, 8, 5")
    end_times_input = st.text_input(
        "Enter the end times of activities (comma-separated)", "2, 4, 6, 7, 9, 9")
    if st.button("Select Activities"):
        start_times = list(map(int, start_times_input.split(',')))
        end_times = list(map(int, end_times_input.split(',')))
        selected_activities = greedy_activity_selection(start_times, end_times)
        st.write("Selected Activities:")
        for activity in selected_activities:
            st.write(
                f"Activity: {activity}, Start Time: {activity[0]}, End Time: {activity[1]}")


elif algorithm_category == "Minimum Spanning Tree":
    st.subheader("🌐 Minimum Spanning Tree (Prim's Algorithm)")
    st.markdown("""
    Given a connected, undirected graph, the minimum spanning tree (MST) is a tree that spans all the vertices in the graph with the minimum possible total edge weight.
    """)
    if st.button("Find Minimum Spanning Tree"):
        mst_edges = minimum_spanning_tree(graph)
        g_original = nx.Graph()
        for frm, edges in graph.items():
            for to, cost in edges.items():
                g_original.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        labels_original = nx.get_edge_attributes(g_original, 'weight')
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_edge_labels(
            g_original, pos_original, edge_labels=labels_original)
        nx.draw_networkx_labels(g_original, pos_original)

        plt.title("Original Graph")
        plt.axis("off")
        st.pyplot(plt)
        g_mst = nx.Graph()
        for frm, to, cost in mst_edges:
            g_mst.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_mst = nx.spring_layout(g_mst)
        labels_mst = nx.get_edge_attributes(g_mst, 'weight')
        nx.draw_networkx_nodes(g_mst, pos_mst)
        nx.draw_networkx_edges(g_mst, pos_mst)
        nx.draw_networkx_edge_labels(g_mst, pos_mst, edge_labels=labels_mst)
        nx.draw_networkx_labels(g_mst, pos_mst)

        plt.title("Minimum Spanning Tree")
        plt.axis("off")
        st.pyplot(plt)
        st.write("Minimum Spanning Tree Edges:")
        for frm, to, cost in mst_edges:
            st.write(f"{frm} - {to}: {cost}")


elif algorithm_category == "Dijkstra's Algorithm":
    st.subheader("🛣️ Dijkstra's Algorithm")
    st.markdown("""
    Dijkstra's algorithm is a graph search algorithm that finds the shortest path between two vertices in a graph.
    """)
    start_vertex = st.selectbox("Choose a starting vertex", vertices)
    if st.button("Find Shortest Paths"):
        distances = dijkstra(graph, start_vertex)
        g_original = nx.Graph()
        for frm, edges in graph.items():
            for to, cost in edges.items():
                g_original.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        labels_original = nx.get_edge_attributes(g_original, 'weight')
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_edge_labels(
            g_original, pos_original, edge_labels=labels_original)
        nx.draw_networkx_labels(g_original, pos_original)

        plt.title("Original Graph")
        plt.axis("off")
        st.pyplot(plt)
        g_shortest_paths = nx.Graph()
        for vertex, distance in distances.items():
            if distance != float('infinity'):
                g_shortest_paths.add_edge(
                    start_vertex, vertex, weight=distance)
        plt.figure()
        pos_shortest_paths = nx.spring_layout(g_shortest_paths)
        labels_shortest_paths = nx.get_edge_attributes(
            g_shortest_paths, 'weight')
        nx.draw_networkx_nodes(g_shortest_paths, pos_shortest_paths)
        nx.draw_networkx_edges(g_shortest_paths, pos_shortest_paths)
        nx.draw_networkx_edge_labels(
            g_shortest_paths, pos_shortest_paths, edge_labels=labels_shortest_paths)
        nx.draw_networkx_labels(g_shortest_paths, pos_shortest_paths)

        plt.title("Shortest Paths from " + start_vertex)
        plt.axis("off")
        st.pyplot(plt)
        st.write("Shortest Paths from " + start_vertex + ":")
        for vertex, distance in distances.items():
            st.write(f"{start_vertex} - {vertex}: {distance}")


elif algorithm_category == "Kruskal's Algorithm":
    st.subheader("🌐 Kruskal's Algorithm")
    st.markdown("""
    Kruskal's algorithm is a greedy algorithm that finds a minimum spanning tree for a connected, undirected graph.
    """)

    if st.button("Find Minimum Spanning Tree (Kruskal's MST)"):
        mst_edges_kruskal = kruskal(graph)
        g_original = nx.Graph()
        for frm, edges in graph.items():
            for to, cost in edges.items():
                g_original.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        labels_original = nx.get_edge_attributes(g_original, 'weight')
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_edge_labels(
            g_original, pos_original, edge_labels=labels_original)
        nx.draw_networkx_labels(g_original, pos_original)
        plt.title("Original Graph")
        plt.axis("off")
        st.pyplot(plt)
        g_mst_kruskal = nx.Graph()
        for frm, to, cost in mst_edges_kruskal:
            g_mst_kruskal.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_mst_kruskal = nx.spring_layout(g_mst_kruskal)
        labels_mst_kruskal = nx.get_edge_attributes(g_mst_kruskal, 'weight')
        nx.draw_networkx_nodes(g_mst_kruskal, pos_mst_kruskal)
        nx.draw_networkx_edges(g_mst_kruskal, pos_mst_kruskal)
        nx.draw_networkx_edge_labels(
            g_mst_kruskal, pos_mst_kruskal, edge_labels=labels_mst_kruskal)
        nx.draw_networkx_labels(g_mst_kruskal, pos_mst_kruskal)

        plt.title("Minimum Spanning Tree (Kruskal's MST)")
        plt.axis("off")
        st.pyplot(plt)
        st.write("Minimum Spanning Tree Edges (Kruskal's MST):")
        for frm, to, cost in mst_edges_kruskal:
            st.write(f"{frm} - {to}: {cost}")


elif algorithm_category == "Prim's Algorithm":
    st.subheader("🌲 Prim's Algorithm")
    st.markdown("""
    Prim's algorithm is a greedy algorithm that finds a minimum spanning tree for a connected, undirected graph.
    """)
    start_vertex_prim = st.selectbox("Choose a starting vertex", vertices)
    if st.button("Find Minimum Spanning Tree (Prim's MST)"):
        mst_edges_prim = prim(graph, start_vertex_prim)
        g_original = nx.Graph()
        for frm, edges in graph.items():
            for to, cost in edges.items():
                g_original.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        labels_original = nx.get_edge_attributes(g_original, 'weight')
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_edge_labels(
            g_original, pos_original, edge_labels=labels_original)
        nx.draw_networkx_labels(g_original, pos_original)

        plt.title("Original Graph")
        plt.axis("off")
        st.pyplot(plt)
        g_mst_prim = nx.Graph()
        for frm, to, cost in mst_edges_prim:
            g_mst_prim.add_edge(frm, to, weight=cost)
        plt.figure()
        pos_mst_prim = nx.spring_layout(g_mst_prim)
        labels_mst_prim = nx.get_edge_attributes(g_mst_prim, 'weight')
        nx.draw_networkx_nodes(g_mst_prim, pos_mst_prim)
        nx.draw_networkx_edges(g_mst_prim, pos_mst_prim)
        nx.draw_networkx_edge_labels(
            g_mst_prim, pos_mst_prim, edge_labels=labels_mst_prim)
        nx.draw_networkx_labels(g_mst_prim, pos_mst_prim)
        plt.title("Minimum Spanning Tree (Prim's MST)")
        plt.axis("off")
        st.pyplot(plt)
        st.write("Minimum Spanning Tree Edges (Prim's MST):")
        for frm, to, cost in mst_edges_prim:
            st.write(f"{frm} - {to}: {cost}")


elif algorithm_category == "Minimum Vertex Cover":
    st.subheader("📎 Minimum Vertex Cover")
    st.markdown("""
    A minimum vertex cover is a set of vertices such that every edge in the graph is incident to at least one vertex in the set.
    """)
    if st.button("Find Minimum Vertex Cover"):
        g_original = nx.Graph()
        for frm, edges in graph.items():
            for to in edges:
                g_original.add_edge(frm, to)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_labels(g_original, pos_original)
        vertex_cover = minimum_vertex_cover(g_original)
        nx.draw_networkx_nodes(
            g_original, pos_original, nodelist=vertex_cover, node_color='red', label='Vertex Cover')
        plt.title("Original Graph with Minimum Vertex Cover")
        plt.axis("off")
        st.pyplot(plt)
        st.write("Minimum Vertex Cover:")
        st.write(vertex_cover)


elif algorithm_category == "Clique Problem":
    st.subheader("🔄 Clique Problem")
    st.markdown("""
    The clique problem is to find a subset of vertices in an undirected graph such that every two distinct vertices in the subset are adjacent.
    """)
    if st.button("Find Maximum Clique"):
        result = []
        vertices = list(graph.keys())
        candidates = set(vertices)
        find_max_clique(graph, [], candidates, result)
        visualize_max_clique(graph, result)
        st.write("Maximum Clique:", result)


elif algorithm_category == "Maximum Independent Set Problem":
    st.subheader("🔄 Maximum Independent Set Problem")
    st.markdown("""
    The Maximum Independent Set (MIS) problem is to find the largest set of vertices in an undirected graph such that no two vertices in the set are adjacent.
    """)
    if st.button("Find Maximum Independent Set"):
        result = []
        vertices = list(graph.keys())
        candidates = set(vertices)
        find_max_independent_set(graph, [], candidates, result)
        visualize_max_independent_set(graph, result)
        st.write("Maximum Independent Set:", result)


elif algorithm_category == "Boolean Satisfiability Problem":
    st.subheader("🔒 Boolean Satisfiability Problem")
    st.markdown("""
    The Boolean Satisfiability Problem (SAT) is to determine whether a given Boolean formula can be made true by assigning Boolean values (true/false) to its variables.
    """)
    formula_input = st.text_input(
        "Enter the Boolean formula (e.g., A or B, not A or C, B or not C):")
    formula = []
    if formula_input:
        for clause_str in formula_input.split(','):
            clause = []
            for literal_str in clause_str.split('or'):
                literal = (literal_str.strip(), False) if 'not' not in literal_str else (
                    literal_str.replace('not', '').strip(), True)
                clause.append(literal)
            formula.append(clause)
        st.write("Original Formula:")
        for clause in formula:
            st.write(clause)
        variables = set(literal[0] for clause in formula for literal in clause)
        assignment = {variable: None for variable in variables}
        if st.button("Find Satisfying Assignment"):
            satisfying_assignment_found = backtracking_satisfy(
                formula, assignment, 0)
            st.write("Satisfying Assignment:", assignment)
            st.write("Satisfiable:", satisfying_assignment_found)


elif algorithm_category == "Hamiltonian Cycle":
    st.subheader("🔄 Hamiltonian Cycle")
    st.markdown("""
    A Hamiltonian cycle is a cycle that visits each vertex exactly once.
    """)
    if st.button("Find Hamiltonian Cycle"):
        g_original = nx.Graph()
        for vertex, neighbors in graph.items():
            for neighbor in neighbors:
                g_original.add_edge(vertex, neighbor)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_labels(g_original, pos_original)
        plt.title("Original Graph")
        plt.axis("off")
        st.pyplot(plt)
        hamiltonian_cycle_result = hamiltonian_cycle(g_original)
        if hamiltonian_cycle_result:
            g_hamiltonian_cycle = nx.Graph()
            for i in range(len(hamiltonian_cycle_result) - 1):
                g_hamiltonian_cycle.add_edge(
                    hamiltonian_cycle_result[i], hamiltonian_cycle_result[i+1])
            plt.figure()
            pos_hamiltonian_cycle = nx.spring_layout(g_hamiltonian_cycle)
            nx.draw_networkx_nodes(g_hamiltonian_cycle, pos_hamiltonian_cycle)
            nx.draw_networkx_edges(g_hamiltonian_cycle, pos_hamiltonian_cycle)
            nx.draw_networkx_labels(g_hamiltonian_cycle, pos_hamiltonian_cycle)
            plt.title("Hamiltonian Cycle")
            plt.axis("off")
            st.pyplot(plt)
            st.write("Hamiltonian Cycle:")
            st.write(" -> ".join(hamiltonian_cycle_result))
        else:
            st.write("No Hamiltonian Cycle found in the given graph.")


elif algorithm_category == "Hamiltonian Path":
    st.subheader("🔄 Hamiltonian Path")
    st.markdown("""
    A Hamiltonian path is a path that visits each vertex exactly once.
    """)
    if st.button("Find Hamiltonian Path"):
        g_original = nx.Graph()
        for vertex, neighbors in graph.items():
            for neighbor in neighbors:
                g_original.add_edge(vertex, neighbor)
        plt.figure()
        pos_original = nx.spring_layout(g_original)
        nx.draw_networkx_nodes(g_original, pos_original)
        nx.draw_networkx_edges(g_original, pos_original)
        nx.draw_networkx_labels(g_original, pos_original)
        plt.title("Original Graph")
        plt.axis("off")
        st.pyplot(plt)
        hamiltonian_path_result = hamiltonian_path(g_original)
        if hamiltonian_path_result:
            g_hamiltonian_path = nx.Graph()
            for i in range(len(hamiltonian_path_result) - 1):
                g_hamiltonian_path.add_edge(
                    hamiltonian_path_result[i], hamiltonian_path_result[i+1])
            plt.figure()
            pos_hamiltonian_path = nx.spring_layout(g_hamiltonian_path)
            nx.draw_networkx_nodes(g_hamiltonian_path, pos_hamiltonian_path)
            nx.draw_networkx_edges(g_hamiltonian_path, pos_hamiltonian_path)
            nx.draw_networkx_labels(g_hamiltonian_path, pos_hamiltonian_path)

            plt.title("Hamiltonian Path")
            plt.axis("off")
            st.pyplot(plt)
            st.write("Hamiltonian Path:")
            st.write(" -> ".join(hamiltonian_path_result))
        else:
            st.write("No Hamiltonian Path found in the given graph.")

    st.markdown("---")
    st.caption("Algorithm Toolbox")
