#!/usr/bin/env python3
from argparse import ArgumentParser
from utils import *


def fwd_pass(user_stfts):
    """
    recordings is the result of split recordings
    returns mean embedding of recordings
    """
    model, *_ = load_saved_model(MODEL_FNAME)

    mean_user_emb = np.zeros((1, 1024))
    for user_stft in user_stfts:
        user_stft = torch.tensor(user_stft)
        out = model.forward_single(user_stft)
        print(out.shape)
        mean_user_emb += out


    mean_user_emb /= len(user_stfts)

    return mean_user_emb

def store_user_embedding(username, emb):
    """
    this function adds username and its emb into database
    emb is mean embedding of the recording returned from fwd_pass
    """
    pass

def get_user_embedding(usernames):
    """
    returns user's emb from the db
    """
    pass

def compare_user_recordings(user1_emb, other_users_emb):
    """
    compare a user's emb with other users emb
    return a list of distances between user1_emb and other_users_emb
    """
    pass

def show_current_users():
    return []

def get_emb():
    record()
    user_stfts = split_recording()
    # print([st.shape for st in user_stfts])
    emb = fwd_pass(user_stfts)
    return emb

def enroll_new_user(username):
    emb = get_emb()
    store_user_embedding(username, emb)

def verify_user(username):
    emb = get_emb()
    user_emb = get_user_embedding(username)
    compare_user_recordings(emb, user_emb)

def identify_user():
    emb = get_emb()
    # Load all user emb
    compare_user_recordings(emb, user_emb)


def main():
    parser = ArgumentParser(description="Speaker Identification and Verification")
    parser.add_argument('-s', '--show-current-users', dest="show",
                        default=False, action="store_true",
                        help="Show current enrolled users")
    parser.add_argument('-e', '--enroll', dest="enroll",
                        default=False, action="store_true",
                        help="Enroll a new user")
    parser.add_argument('-v', '--verify', dest="verify",
                        default=False, action="store_true",
                        help="Verify a user from the ones in the database")
    parser.add_argument('-i', '--identify', dest="identify",
                        default=False, action="store_true",
                        help="Identify a user")
    parser.add_argument('-u', '--username', type=str, default=None,
                        help="Name of the user to enroll or verify")

    args = parser.parse_args()

    if args.show:
        show_current_users()

    elif args.enroll:
        username = args.username
        assert username is not None, "Enter username"
        assert username not in show_current_users(), "Username already exists in database"
        enroll_new_user(username)

    elif args.verify:
        username = args.username
        assert username is not None, "Enter username"
        assert username in show_current_users(), "Unrecognized username"
        verify_user(username)


    elif args.identify:
        identify_user()

    else:
        show_current_users()


if __name__ == "__main__":
    main()
