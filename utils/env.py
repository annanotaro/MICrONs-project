from caveclient import CAVEclient
client = CAVEclient("minnie65_public")
print("Success")
print(client.auth.token)