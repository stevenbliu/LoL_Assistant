import socket
import os


try:
    ip = socket.gethostbyname("na1.api.riotgames.com")
    print(f"Resolved na1.api.riotgames.com to {ip}")
except Exception as e:
    print("DNS resolution failed:", e)

# summoner_name = input("Enter summoner name: ")
summoner_name = "me"
PUUID = "h_lHGPIV0_8WIWXNBz_vfXhRYwmVnQYm3Asr6TMFrYUpeyLeQEscs42olQGaaBhxgvFAsEj0lTTIdA"
print(f"Fetching data for summoner: {summoner_name}")
