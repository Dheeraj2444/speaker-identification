#!/usr/bin/env python3
from argparse import ArgumentParser
from utils import *


def fwd_pass(recordings, model_fname):
    """
    recordings is the result of split recordings
    returns mean embedding of recordings
    """
    pass

def store_user_embedding(username, emb):
    """
    this function adds username and its emb into database
    emb is mean embedding of the recording returned from fwd_pass
    """
    pass

def get_user_embedding(username):
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

def enroll_new_user(username):
    record()
    user_stfts = split_recording()
    print([st.shape for st in user_stfts])

def verify_user():
    pass

def identify_user():
    pass


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
