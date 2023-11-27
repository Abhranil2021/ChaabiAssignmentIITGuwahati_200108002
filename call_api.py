import requests

while(1):
    print("Please Enter query:\n")
    query = input()
    print("Generating Answer ...", end=" ")
    res = requests.get('http://localhost:5000/query-engine?query='+str(query))
    print("Done")
    print(res.json())
    print(" ----------------------------------------------\n")