def agent_position_targets(next_tiles):
    """
    next_tiles: [B, 10, 10]

    return:
    target index in [0, 99] for the agent position
    """

    batch_size = next_tiles.shape[0]
    flat = next_tiles.view(batch_size, -1)
    return (flat == 4).float().argmax(dim=1)


def single_agent_rate(tile_classes):
    agent_counts = (tile_classes == 4).sum(dim=(1, 2))
    return (agent_counts == 1).float().mean().item()


def find_single_agent(tile_classes):
    """
    tile_classes: [10, 10]

    Returns:
    (row, col) if exactly one agent exists, otherwise None
    """
    
    positions = (tile_classes == 4).nonzero(as_tuple=False)
    

    if positions.shape[0] != 1:
        return None
    
    return tuple(positions[0].tolist())