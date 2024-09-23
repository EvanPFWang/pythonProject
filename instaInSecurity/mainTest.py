import instaloader
import time
import logging

# Setup logging
logging.basicConfig(filename='insta_script_test.log', level=logging.INFO)

# Initialize Instaloader instances
L_main = instaloader.Instaloader()
L_finsta = instaloader.Instaloader()

# Login to your main account
main_username = 'your_main_username'
main_password = 'your_main_password'
try:
    L_main.login(main_username, main_password)
    print("Main account login successful.")
    logging.info("Main account login successful.")
except instaloader.exceptions.InstaloaderException as e:
    print(f"Failed to log into main account: {e}")
    logging.error(f"Failed to log into main account: {e}")

# Login to your finsta account
finsta_username = 'your_finsta_username'
finsta_password = 'your_finsta_password'
try:
    L_finsta.login(finsta_username, finsta_password)
    print("Finsta account login successful.")
    logging.info("Finsta account login successful.")
except instaloader.exceptions.InstaloaderException as e:
    print(f"Failed to log into finsta account: {e}")
    logging.error(f"Failed to log into finsta account: {e}")

# Load profiles for both accounts
profile_main = instaloader.Profile.from_username(L_main.context, main_username)
profile_finsta = instaloader.Profile.from_username(L_finsta.context, finsta_username)

# Fetch the list of accounts you follow on your main account
try:
    following_main = set(profile_main.get_followees())
    print(f"Successfully fetched followees for main account. Total: {len(following_main)}")
    logging.info(f"Successfully fetched followees for main account. Total: {len(following_main)}")
except instaloader.exceptions.InstaloaderException as e:
    print(f"Error fetching followees for main account: {e}")
    logging.error(f"Error fetching followees for main account: {e}")

# Fetch the list of accounts that follow your main account back
try:
    followers_main = set(profile_main.get_followers())
    print(f"Successfully fetched followers for main account. Total: {len(followers_main)}")
    logging.info(f"Successfully fetched followers for main account. Total: {len(followers_main)}")
except instaloader.exceptions.InstaloaderException as e:
    print(f"Error fetching followers for main account: {e}")
    logging.error(f"Error fetching followers for main account: {e}")

# Find accounts you follow but don't follow you back
non_follow_back = following_main - followers_main

# Filter accounts with more than 5k followers
non_follow_back_with_5k_followers = [acc for acc in non_follow_back if acc.followers > 5000]
if not non_follow_back_with_5k_followers:
    print("No accounts found with more than 5k followers who don't follow you back.")
    logging.info("No accounts found with more than 5k followers who don't follow you back.")
else:
    print(f"Accounts found: {len(non_follow_back_with_5k_followers)}")
    logging.info(f"Accounts found: {len(non_follow_back_with_5k_followers)}")

# Simulate follow/unfollow operations for testing
for acc in non_follow_back_with_5k_followers:
    try:
        # Simulate following with finsta
        print(f"Would follow {acc.username} with finsta...")
        logging.info(f"Would follow {acc.username} with finsta...")
        time.sleep(2)  # Pause to avoid rate limits

        # Simulate unfollowing from main
        print(f"Would unfollow {acc.username} from main account...")
        logging.info(f"Would unfollow {acc.username} from main account...")
        time.sleep(2)  # Pause to avoid rate limits

    except instaloader.exceptions.InstaloaderException as e:
        print(f"Error handling {acc.username}: {e}")
        logging.error(f"Error handling {acc.username}: {e}")
    except Exception as e:
        print(f"General error occurred for {acc.username}: {e}")
        logging.error(f"General error occurred for {acc.username}: {e}")

print("Finished testing follow and unfollow operations.")
logging.info("Finished testing follow and unfollow operations.")
