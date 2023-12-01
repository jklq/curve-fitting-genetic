"""

Node in tree:
- Value
- Children

value:
"*" - times
"+" - plus
"-" - minus
[number] - number
"x" - x variable

"""

import numpy as np

operators = ["*", "+", "-"]
variables = ["x"]

seed = 1231
rng = np.random.default_rng(seed)

targetTree = [
    "*", [
        ["*",
            [
                ["x", None],
                ["+", [
                    ["x", None],
                    [1, None]
                ]]
            ]
         ],
        ["-",
            [
                ["x", None],
                ["4", None]
            ]
         ],
    ]
]


def performAction(action, x, a, b):
    if a == "x":
        a = x
    if b == "x":
        b = x

    a, b = float(a), float(b)

    if action == "*":
        return a * b
    elif action == "+":
        return a + b
    elif action == "-":
        return a - b
    elif action == "**":
        return a ** b


def calculateNode(node, x, first_time=True):
    action, children = node[0], node[1]

    if first_time and children == None:
        return x if action == "x" else action

    left, right = children

    left_value = left[0] if left[1] is None else calculateNode(
        left, x, first_time=False)

    right_value = right[0] if right[1] is None else calculateNode(
        right, x, first_time=False)

    return performAction(action, x, left_value, right_value)


def getRandomNodeValue(no_children=False):

    nodeValueChoices = [float(rng.integers(low=-1000, high=1001) / 10),
                        np.random.choice(variables)]

    if not no_children:
        nodeValueChoices = np.append(nodeValueChoices,
                                     np.random.choice(operators))

    choice = np.random.choice(nodeValueChoices)
    return choice, choice in operators


def generateRandomTree(depth=0, max_depth=4):
    no_children = False
    if depth == max_depth:
        no_children = True

    node_value, has_children = getRandomNodeValue(no_children)

    return [node_value, [generateRandomTree(depth+1), generateRandomTree(depth+1)] if has_children else None]


def calculateLoss(X, answers, predictionModel):
    predictions = []
    loss = 0

    for i, _ in enumerate(answers):
        predictions.append(calculateNode(predictionModel, X[i]))

    for set in zip(answers, predictions):
        answer, prediction = set

        loss += (float(answer) - float(prediction)) ** 2

    return loss / len(predictions)


def listTreeNodeCoords(tree):

    def listTreeNodesHelper(tree, treeList=[[]], depth=0):
        if tree[1] == None:
            return treeList

        left, right = tree[1]

        return np.append(
            listTreeNodesHelper(left, treeList=treeList +
                                [treeList[-1] + [0]], depth=depth+1),
            listTreeNodesHelper(right, treeList=treeList +
                                [treeList[-1] + [1]], depth=depth+1),
        )

    coords = listTreeNodesHelper(tree)
    if len(coords) == 1 and len(coords[0]) == 0:
        return []
    return np.delete(np.unique(coords), 0)


def getNodeFromCoords(tree, coords):
    for index in coords:
        tree = tree[1][index]
    return tree


def replaceNodeOnTree(tree, node, coords):
    if not coords:
        return node

    value, children = tree
    new_children = children.copy() if children else None

    if children:
        direction = coords[0]
        new_children[direction] = replaceNodeOnTree(
            children[direction], node, coords[1:])

    return [value, new_children]


def getRandomCoordsOfTree(tree):
    nodeCoordsList = listTreeNodeCoords(tree)
    if len(nodeCoordsList) == 0:
        return []
    treeCoords = np.random.choice(nodeCoordsList)
    return treeCoords


def getRandomNodeFromTree(tree):
    nodeCoordsList = listTreeNodeCoords(tree)
    if len(nodeCoordsList) == 0:
        return tree
    treeCoords = np.random.choice(nodeCoordsList)
    return getNodeFromCoords(tree, treeCoords)


# Given models sorted from fittest to least fit
# Returns a list of new population of (largely crossed and mutated) models
def crossAndMutate(modelPopulation):
    population = []
    probabilities = np.linspace(1, 0, 100)
    probabilities /= probabilities.sum()

    n = len(modelPopulation)

    for i in range(n):
        # cross
        (_, donating), (_, receiving) = [np.random.choice(
            modelPopulation[:100], p=probabilities) for i in range(0, 2)]

        receiving = replaceNodeOnTree(
            receiving, getRandomNodeFromTree(donating), getRandomCoordsOfTree(receiving))

        # TODO: the max depth is not respected. depth=2 set arbitrarily
        # mutate
        receiving = replaceNodeOnTree(
            receiving, generateRandomTree(
                depth=2), getRandomCoordsOfTree(receiving)
        )
        # add to population
        population.append(receiving)
    return population


def tree_to_equation(tree):
    """
    Converts a tree model into a human-readable infix mathematical expression.
    """
    if tree is None or not tree:
        return ""

    # Handling basic number or variable case
    if not isinstance(tree, list) or len(tree) == 2 and tree[1] is None:
        return str(tree[0])

    # Extracting the operator and its operands
    operator, operands = tree

    # Converting each operand recursively and adding brackets for clarity
    converted_operands = [
        f"({tree_to_equation(operand)})" for operand in operands]

    # Formatting expressions based on the operator
    if operator == '+':
        return ' + '.join(converted_operands)
    elif operator == '-':
        return ' - '.join(converted_operands)
    elif operator == '*':
        return ' * '.join(converted_operands)
    else:
        # For unsupported operators or formats
        return "Unsupported Format"


# Parameters
X = np.arange(-100, 100, 0.5)
initial_population = 1000
num_generations = 100  # Specify the number of generations here

# Calculate the answers for each value in X using the target tree
answers = np.array([calculateNode(targetTree, x) for x in X])


def run_generation(models, X, answers):
    # Calculate losses for the models
    losses = [calculateLoss(X, answers, model) for model in models]

    # Zip and sort models based on loss
    modelLosses = np.array(list(zip(losses, models)), dtype=[
                           ("loss", float), ("model", object)])
    modelLossesSorted = np.sort(modelLosses, order='loss')

    # Generate next generation
    next_gen = crossAndMutate(modelLossesSorted)

    return next_gen, modelLossesSorted


# Initialize first generation of models
current_generation = [generateRandomTree() for _ in range(initial_population)]

# Run the simulation for specified number of generations

print(f"Target equation: {tree_to_equation(targetTree)}")

for gen in range(num_generations):
    current_generation, sorted_models = run_generation(
        current_generation, X, answers)
    print(
        f"Generation {gen + 1} - Top model loss: {sorted_models[0]['loss']:.4f}")
    if sorted_models[0]['loss'] == 0:
        break

# Display final generation's top models
print(f"Total models in the final generation: {len(current_generation)}")
print("Top models from the final generation based on loss:")
for i, (loss, model) in enumerate(sorted_models[:5]):  # Display top 5 models
    print(f"Rank {i+1}, Loss: {loss:.4f}, Model: {tree_to_equation(model)}")
