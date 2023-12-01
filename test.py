def replaceNodeOnTree(tree, node, coords):
    coords.pop(0)
    # Base case: if coords is empty, return the new node.
    if not coords:
        return node

    # Copy the current node to avoid modifying the original tree.
    value, children = tree
    new_children = children.copy() if children else None

    # Recursive step: navigate the tree according to the first coordinate.
    if children:
        direction = coords[0]
        new_children[direction] = replaceNodeOnTree(
            children[direction], node, coords[1:])

    # Return the new tree with the replaced node.
    return [value, new_children]


# Example usage
input_tree = ['*', [['+', [['x', None], [1, None]]],
                    ['-', [['x', None], [1, None]]]]]
node_to_insert = ['x', None]
coords = [0, 1]
output_tree = replaceNodeOnTree(input_tree, node_to_insert, coords)
print(output_tree)
