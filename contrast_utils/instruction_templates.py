
def get_objects_from_instruction(instruction, get_all_parts=False):
    # if instruction == 'Use right gripper to open top drawer, pick up red square cube, place it inside top drawer, and then close drawer.':
    print(instruction)
    if 'red square block' in instruction and 'drawer' in instruction:
        return ["drawer1", "drawer2", "drawer3", "red block"]
    elif 'banana' in instruction and 'pot' in instruction and 'lid' in instruction:
        return ["banana", "pot", "lid"]
    elif 'two pins' in instruction and 'jig' in instruction:
        return ["pin1", "pin2", 'jig']
    else:
        raise ValueError(f"Instruction not supported: {instruction}")

