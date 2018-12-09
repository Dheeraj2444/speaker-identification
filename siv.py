#!/usr/bin/env python3
from argparse import ArgumentParser
from utils import *
from network import *


def fwd_pass(user_stfts):
    """
    recordings is the result of split recordings
    returns mean embedding of recordings
    """
    model, *_ = load_saved_model(MODEL_FNAME)

    mean_user_emb = torch.ones(1, 1024)
    for user_stft in user_stfts:
        user_stft = torch.tensor(user_stft).to(device)
        out = model.forward_single(user_stft)
        print(out.shape)
        mean_user_emb += out

    mean_user_emb /= len(user_stfts)

    return mean_user_emb.detach().cpu().numpy()


def store_user_embedding(username, emb):
    """
    this function adds username and its emb into database
    emb is mean embedding of the recording returned from fwd_pass
    """
    speaker_models = load_speaker_models()
    speaker_models[username] = emb
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)
    print("Successfully added user {} to database".format(username))


def get_user_embedding(usernames):
    """
    returns list of users emb from the db
    """
    speaker_models = load_speaker_models()
    return [speaker_models[username] for username in usernames]


def load_speaker_models():
    if not os.path.exists(SPEAKER_MODELS_FILE):
        return dict()

    with open(SPEAKER_MODELS_FILE, 'rb') as fhand:
        speaker_models = pickle.load(fhand)

    return speaker_models


def show_current_users():
    speaker_models = load_speaker_models()
    return list(speaker_models.keys())


def get_emb():
    record()
    user_stfts = split_recording()
    emb = fwd_pass(user_stfts)
    return emb


def enroll_new_user(username):
    emb = get_emb()
    store_user_embedding(username, emb)


def verify_user(username):
    emb = get_emb()
    user_emb = get_user_embedding(username)
    dist = scipy.spatial.distance.cdist(emb, user_emb, DISTANCE_METRIC)
    print(dist)
    return dist < THRESHOLD


def identify_user():
    emb = get_emb()
    speaker_models = load_speaker_models()
    dist = [(other_user, scipy.spatial.distance.cdist(emb, speaker_model[other_user],
                              DISTANCE_METRIC)) for other_user in skeaker_models]
    print(dist)
    username, min_dist = min(dist, key=lambda x:x[1])

    if min_dist < THRESHOLD:
        return username
    return None


def delete_user(username):
    speaker_models = load_speaker_models()
    _ = speaker_models.pop(username)
    print("Successfully removed {} from databse".format(username))

def clear_database():
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(dict(), fhand)


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
    parser.add_argument('-d', '--delete', dest="delete",
                        default=False, action="store_true",
                        help="Delete user from database")
    parser.add_argument('-c', '--clear', dest="clear",
                        default=False, action="store_true",
                        help="Clear Database")
    parser.add_argument('-u', '--username', type=str, default=None,
                        help="Name of the user to enroll or verify")

    args = parser.parse_args()

    if args.show:
        print(show_current_users())

    elif args.enroll:
        username = args.username
        assert username is not None, "Enter username"
        assert username not in show_current_users(), "Username already exists in database"
        enroll_new_user(username)

    elif args.verify:
        username = args.username
        assert username is not None, "Enter username"
        assert username in show_current_users(), "Unrecognized username"
        if verify_user(username):
            print("User verified")
        else:
            print("Unknown user")

    elif args.identify:
        identified_user = identify_user()
        print("Identified User {}".format(identified_user))

    elif args.delete:
        username = args.username
        assert username is not None, "Enter username"
        assert username in show_current_users(), "Unrecognized username"
        delete_user(username)

    elif args.clear:
        clear_database()

    else:
        show_current_users()


if __name__ == "__main__":
    main()
