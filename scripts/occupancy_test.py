from occupancy import calculator

test_case = [
    [128, 64, 4096, "7.0", 50],
    [256, 64, 4096, "7.0", 50],
    [128, 32, 4096, "7.0", 50],
    [128, 32, 2048, "7.0", 100]
]

for test in test_case:
    output = int(calculator(test[0], test[1], test[2], test[3]) * 100)
    print(output, test[4])
