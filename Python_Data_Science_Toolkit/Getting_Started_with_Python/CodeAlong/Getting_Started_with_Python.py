import yaml

print("#" * 50)
print("IPL Dataset Analysis")
print("#" * 50)

# using with open command to read the file
with open('ipl_match.yaml') as f:
    data = yaml.load(f)
print(type(data))

print("#" * 50)
print("In which city the match was played and where was it played")
print("#" * 50)
print(data.get('info').get('city'))
print(data.get('info').get('venue'))

print("#" * 50)
print("Which are all the teams that played in the tournament ? How many teams participated in total?")
print("#" * 50)
print(data.get('info').get('teams'))
print(len(data.get('info').get('teams')))

print("#" * 50)
print("Which team won the toss and what was the decision of toss winner")
print("#" * 50)
print(data.get('info').get('toss').get('winner'))
print(data.get('info').get('toss').get('decision'))

print("#" * 50)
print("Find the first bowler who played the first ball of the first inning. Also the first batsman who faced first delivery ?")
print("#" * 50)
print(data.get('innings')[0].get('1st innings').get('deliveries')[0].get(0.1).get('bowler'))
print(data.get('innings')[0].get('1st innings').get('deliveries')[0].get(0.1).get('batsman'))

print("#" * 50)
print("How many deliveries were delivered in first inning")
print("#" * 50)
print(len(data.get('innings')[0].get('1st innings').get('deliveries')))

print("#" * 50)
print("How many deliveries were delivered in second inning")
print("#" * 50)
print(len(data.get('innings')[1].get('2nd innings').get('deliveries')))

print("#" * 50)
print("Which team won and how ?")
print("#" * 50)
print(data.get('info').get('outcome').get('winner'))
print(data.get('info').get('outcome').get('by').get('runs'))