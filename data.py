items = ['pine', 'oak', 'rose', 'daisy',
         'robin', 'canary', 'sunfish', 'salmon']
relations = ['ISA', 'is', 'can', 'has']
attributes = ['living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'pine', 'oak', 'rose', 'daisy', 'robin', 'canary', 'sunfish', 'salmon', 'pretty',
              'tall', 'living', 'green', 'red', 'yellow', 'grow', 'move', 'swim', 'fly', 'sing', 'bark', 'petals', 'wings', 'feathers', 'scales', 'gills', 'roots', 'skin']

data = [['pine', 'ISA', ['tree', 'plant', 'pine', 'living thing']],
        ['pine', 'has', ['bark', 'roots']], ['pine', 'can', ['grow']], [
            'pine', 'is', ['tall', 'living', 'green']],
        ['oak', 'ISA', ['living thing', 'tree', 'plant', 'oak']],
        ['oak', 'is', ['tall', 'living', 'green']],
        ['oak', 'can', ['grow']],
        ['oak', 'has', ['bark', 'roots']],
        ['rose', 'ISA', ['flower', 'living thing', 'plant', 'rose']],
        ['rose', 'is', ['pretty', 'red', 'living']],
        ['rose', 'can', ['grow']],
        ['rose', 'has', ['petals', 'roots']],
        ['daisy', 'ISA', ['living thing', 'plant', 'flower', 'daisy']],
        ['daisy', 'is', ['pretty', 'yellow', 'living']],
        ['daisy', 'can', ['grow']],
        ['daisy', 'has', ['petals', 'roots']],
        ['robin', 'ISA', ['living thing', 'animal', 'bird', 'robin']],
        ['robin', 'is', ['living', 'red']],
        ['robin', 'can', ['grow', 'move', 'fly', 'sing']],
        ['robin', 'has', ['wings', 'feathers', 'skin']],
        ['canary', 'ISA', ['living thing', 'animal', 'bird', 'canary']],
        ['canary', 'is', ['living', 'yellow']],
        ['canary', 'can', ['grow', 'move', 'fly', 'sing']],
        ['canary', 'has', ['wings', 'feathers', 'skin']],
        ['sunfish', 'ISA', ['living thing', 'animal', 'fish', 'sunfish']],
        ['sunfish', 'is', ['living']],
        ['sunfish', 'can', ['grow', 'swim', 'move']],
        ['sunfish', 'has', ['scales', 'gills', 'skin']],
        ['salmon', 'ISA', ['living thing', 'animal', 'fish', 'salmon']],
        ['salmon', 'is', ['pretty', 'living']],
        ['salmon', 'can', ['swim', 'grow', 'move']],
        ['salmon', 'has', ['scales', 'skin', 'gills']]
        ]
