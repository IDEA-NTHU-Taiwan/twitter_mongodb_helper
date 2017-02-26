# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides methods to query the MongoDB instance
"""

import os
import re
import string
from time import time
import cProfile
import glob
import csv
from pprint import pprint
import pstats
import requests
from itertools import chain
from joblib import Parallel, delayed, cpu_count
import pymongo
import ujson
from bson.son import SON
from bson.code import Code
from bson.objectid import ObjectId
from nltk.util import ngrams
from nltk.corpus import stopwords

JSON_PATH = ""
CSV_PATH = ""
DATA_PATH = "data/"
DB_URL = os.environ["MONGODB_URL"]
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
USER_MENTIONS_LIMIT = 50
PUNCTUATION = list(string.punctuation)
STOP_LIST = dict.fromkeys(stopwords.words(
    "english") + PUNCTUATION + ["rt", "via", "RT"])


def timing(func):
    """Decorator for timing run time of a function
    """
    def wrap(*args):
        """Wrapper
        """
        time1 = time()
        ret = func(*args)
        time2 = time()
        print '%s function took %0.3f ms' % (func.func_name, (time2 - time1) * 1000.0)
        return ret
    return wrap


def do_cprofile(func):
    """Decorator for profiling a function
    """

    def profiled_func(*args, **kwargs):
        """Wrapper
        """
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            stats = pstats.Stats(profile)
            stats.sort_stats("time").print_stats(20)

    return profiled_func


def connect():
    """Initializes a pymongo conection object.

    Returns:
        pymongo.MongoClient: Connection object for Mongo DB_URL
    """
    try:
        conn = pymongo.MongoClient(DB_URL)
        print "Connected to DB at " + DB_URL + " successfully"
    except pymongo.errors.ConnectionFailure, ex:
        print "Could not connect to MongoDB: %s" % ex
    return conn


def send_job_completion(run_time, args):
    """Format and print the details of a completed job

    Args:
        run_time (list): Start and end times.
        args (list): Contains the following:
            0: function_name (str): Name of the function that was run.
            1: message_text  (str): Text to be sent in notification.
    """

    time_diff = round((run_time[1] - run_time[0]), 2)
    print "%s function took %0.3f seconds" % (args[0], time_diff)
    send_notification = send_job_notification(
        "Lab Computer: " + ": " + args[1] + " took " + str(time_diff) + " seconds", "Complete")
    print send_notification.content


def send_job_notification(title, body):
    """ Send a notification via Pushbullet.

     Args:
        json_obj (json_obj).

    Indicates whether a job has completed or whether an error occured.
    """
    headers = {"Access-Token": 'PUSHBULLET_API_KEY',
               "Content-Type": "application/json"}
    payload = {"type": "note", "title": title, "body": ujson.dumps(body)}
    url = "https://api.pushbullet.com/v2/pushes"
    return requests.post(url, headers=headers, data=ujson.dumps(payload))


def get_filenames():
    """Reads all the json files in the folder and removes the drive and path and
    extension, only returning a list of strings with the file names.

    Returns:
        list: List of plain filenames
    """
    file_path = glob.glob(JSON_PATH + "*.json")
    result = []
    for entry in file_path:
        _, path = os.path.splitdrive(entry)
        path, filename = os.path.split(path)
        name = os.path.splitext(filename)[0]
        result.append(str(name))
    return result


def read_json_file(filename, path):
    """Accepts a file name and loads it as a json object.
    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.
    Returns:
        obj: json object
    """
    result = []
    try:
        with open(path + filename + ".json", "r") as entry:
            result = ujson.load(entry)
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_json_file(filename, path, result):
    """Writes the result to json with the given filename.

    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """
    with open(path + filename + ".json", "w+") as json_file:
        ujson.dump(result, json_file)
    json_file.close()


def read_csv_file(filename, path):
    """Accepts a file name and loads it as a list.
    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.

    Returns:
        list: List of strings.
    """

    try:
        with open(path + filename + '.csv', 'r') as entry:
            reader = csv.reader(entry)
            temp = list(reader)
            # flatten to 1D, it gets loaded as 2D array
            result = [x for sublist in temp for x in sublist]
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_csv_file(filename, result, path):
    """Writes the result to csv with the given filename.
    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    output = open(path + filename + '.csv', 'wb')
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator='\n')
    for val in result:
        writer.writerow([val])
    # Print one a single row
    # writer.writerow(result)


def count_entries(file_list):
    """Performs a count of the number of number of words in the corpus
     Args:
        file_list  (list): list of file names.

    Returns:
        list: A list of json objects containing the count per file name
    """
    result = []
    for obj in file_list:
        with open(CSV_PATH + obj + '.csv', "r") as entry:
            reader = csv.reader(entry, delimiter=",")
            col_count = len(reader.next())
            res = {"Filename": obj, "Count": col_count}
            result.append(res)
    return result


def build_query_string(query_words):
    """Builds an OR concatenated string for querying the Twitter Search API.
    Args:
        query_words (list): list of words to be concatenated.

    Returns:
        list: List of words concatenated with OR.
    """
    result = ''.join(
        [q + ' OR ' for q in query_words[0:(len(query_words) - 1)]])
    return result + str(query_words[len(query_words) - 1])


def test_file_operations():
    """
    Test previous methods
    """
    file_list = get_filenames()
    num_entries = count_entries(file_list)
    pprint(num_entries)
    res = read_csv_file('csvname', CSV_PATH)
    res2 = build_query_string(res)
    print res2


def unicode_to_utf(unicode_list):
    """ Converts a list of strings from unicode to utf8
    Args:
        unicode_list (list): A list of unicode strings.

    Returns:
        list: UTF8 converted list of strings.
    """
    return [x.encode('UTF8') for x in unicode_list]


def get_language_list(client, db_name):
    """Returns a list of all the matching languages within the collection
     Args:
        client  (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name (str): Name of database to query.

    Returns:
        list: List of languages within the twitter collection.
    """
    dbo = client[db_name]
    distinct_lang = dbo.tweets.distinct("lang")
    return unicode_to_utf(distinct_lang)


def get_language_distribution(client, db_name, lang_list):
    """Returns the distribution of tweets matching either
    english, undefined or spanish.

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.

    Returns:
        list: Distribution for each language in lang_list.
    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
        {"$project": {"language": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("language", -1)])}
    ]
    return dbo.tweets.aggregate(pipeline)


def test_get_language_distribution(client):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = get_language_list(client, 'twitter')
    cursor = get_language_distribution(client, 'twitter', lang_list)
    write_json_file('language_distribution', DATA_PATH, list(cursor))


def test_get_language_subset(client):
    """Test and print the results of aggregation
    Constrains language list to en, und, es.

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = ['en', 'und', 'es']
    cursor = get_language_distribution(client, 'twitter', lang_list)
    for document in cursor:
        print document


def create_lang_subset(client, db_name, lang):
    """Subsets the collection by the specified language.
    Outputs value to new collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang        (list): language to match on.

    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": lang}},
        {"$out": "subset_" + lang}
    ]
    dbo.tweets.aggregate(pipeline)


def get_top_k_users(client, db_name, lang_list, k_filter):
    """Finds the top k users in the collection.
    k_filter is the name of an array in the collection, we apply the $unwind operator to it

     Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        k_filter    (str):  Name of an array in the collection.abs

    Returns:
        list: List of objects containing id_str, screen_name and the frequency of appearance.
    """
    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": {"id_str": k_filter + ".id_str", "screen_name":
                            k_filter + ".screen_name"}, "count": {"$sum": 1}}},
        {"$project": {"id_str": "$_id.id_str",
                      "screen_name": "$_id.screen_name", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("id_str", -1)])}
    ]
    return dbo.tweets.aggregate(pipeline, allowDiskUse=True)


def get_top_k_hashtags(client, db_name, lang_list, k_filter, limit, k_value):
    """Finds the top k hashtags in the collection.
    k_filter is the name of an array in the collection, we apply the $unwind operator to it

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        k_filter    (str):  Name of an array in the collection.abs
        limit       (int):  Limit for the number of results to return.
        k_value     (int):  Filter for the number of occurences for each hashtag

    Returns:
        list: List of objects containing _id, hashtag text and the frequency of appearance.
    """

    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": k_filter + ".text", "count": {"$sum": 1}}},
        {"$project": {"hashtag": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("_id", -1)])},
        {"$match": {"count": {"$gt": k_value}}},
        {"$limit": limit},
    ]
    return dbo.tweets.aggregate(pipeline)


@timing
def test_get_top_k_users(client, db_name, lang_list, k_filter):
    """Test and print results of top k aggregation
    """
    cursor = get_top_k_users(client, db_name, lang_list,
                             k_filter, USER_MENTIONS_LIMIT)
   # Write directly to json file
    write_json_file('user_distribution', DATA_PATH, list(cursor))


@do_cprofile
def test_get_top_k_hashtags(client, db_name, lang_list, k_filter, k_value):
    """Test and print results of top k aggregation
    """
    frequency = []
    cursor = get_top_k_hashtags(
        client, db_name, lang_list, k_filter, HASHTAG_LIMIT, k_value)
    for document in cursor:
        frequency.append({'hashtag': document['_id'],
                          'value': document['count']})
    pprint(frequency)
    write_json_file('hashtag_distribution', DATA_PATH, frequency)


def user_mentions_map_reduce(client, db_name, subset, output_name):
    """Map reduce that returns the number of times a user is mentioned

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """
    map_function = Code("function () {"
                        "    var userMentions = this.entities.user_mentions;"
                        "    for (var i = 0; i < userMentions.length; i ++){"
                        "        if (userMentions[i].screen_name.length > 0) {"
                        "            emit (userMentions[i].screen_name, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyUsername, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    frequency = []
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name)

    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})

    frequency = sorted(frequency, key=lambda k: k['value'], reverse=True)
    write_json_file('user_distribution_mr', DATA_PATH, frequency)
    pprint(frequency)


def hashtag_map_reduce(client, db_name, subset, output_name):
    """Map reduce that returns the number of times a hashtag is used

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """
    map_function = Code("function () {"
                        "    var hashtags = this.entities.hashtags;"
                        "    for (var i = 0; i < hashtags.length; i ++){"
                        "        if (hashtags[i].text.length > 0) {"
                        "            emit (hashtags[i].text, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyHashtag, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    frequency = []
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name)

    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})

    frequency = sorted(frequency, key=lambda k: k['value'], reverse=True)
    write_json_file('hashtag_distribution_mr', DATA_PATH, frequency)
    pprint(frequency)


def find_by_object_id(client, db_name, subset, object_id):
    """Fetches the specified object from the specified collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
        object_id   (str): Object ID to fetch.
    """
    dbo = client[db_name]
    cursor = dbo[subset].find({"_id": ObjectId(object_id)})
    pprint(cursor["_id"])


def get_hashtag_collection(client, db_name, subset):
    """Fetches the specified hashtag collection and writes it to a json file

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
    """
    dbo = client[db_name]
    cursor = dbo[subset].find({"count": {"$gt": 500}}, {
                              "hashtag": 1, "count": 1, "_id": 0})
    write_json_file(subset, DATA_PATH, list(cursor))


def finder(client, db_name, subset, k_items):
    """Fetches k obects from the specified collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
        k_items     (int): Number of items to retrieve.
    """
    dbo = client[db_name]
    cursor = dbo[subset].find().limit(k_items)
    for document in cursor:
        pprint(document)
        pprint(str(document["_id"]))


@do_cprofile
def create_ngrams(text_list, length):
    """ Create ngrams of the specified length from a string of text
    Args:
        text_list   (list): Pre-tokenized text to process.
        length      (int):  Length of ngrams to create.
    """

    clean_tokens = [token for token in text_list if token not in PUNCTUATION]
    return [" ".join(i for i in ngram) for ngram in ngrams(clean_tokens, length)]


def remove_urls(raw_text):
    """ Removes urls from text

    Args:
        raw_text (str): Text to filter.
    """
    return re.sub(
        r"(?:http|https):\/\/((?:[\w-]+)(?:\.[\w-]+)+)(?:[\w.,@?^=%&amp;:\/~+#-]*[\w@?^=%&amp;\/~+#-])?", "", raw_text)


@timing
def test_linear_scan(connection_params, sample_size):
    """Test linear scan
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        sample_size (int): Number of documents to retrieve.
    """
    client = connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    cursor = dbo[collection].find({}, {"_id": 1}).limit(sample_size)
    documents = {str(document["_id"]) for document in cursor}
    pprint(len(documents))


def process_partition(partition, connection_params):
    """ Thread safe process
    partition stores a tuple with the skip and limit values
    Args:
        partition   (tuple): Contains skip and limit values.
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """
    client = connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    cursor = dbo[collection].find({}, {"_id": 1}).skip(
        partition[0]).limit(partition[1])
    documents = {str(document["_id"]) for document in cursor}
    return documents


@timing
def parallel_test(num_cores, connection_params, sample_size):
    """Test parallel functionality
    Args:
        num_cores (int): Number of processes to use, usually associated with the number of cores
        your processor has, check this with cpu_count().
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        sample_size (int): Number of documents to retrieve.
    """

    # Here, we divide the collection in k slices.
    partition_size = sample_size // num_cores

    # We need to set the start and end indices for the collection we want to access in parallel
    # Ex. collection_size = 2001, num_cores = 4, the list comprehension
    # below gives the following skip:limit indices:
    # (0, 500), (500, 500), (1000, 500), (1500, 500), (2000, 500)

    # We then deal with the last index since it exceeds the size of the
    # collection
    partitions = [(i, partition_size)
                  for i in range(0, sample_size, partition_size)]

    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (sample_size - partitions[-1][0]))

    # See https://pythonhosted.org/joblib/parallel.html for more details
    # The library allows us to pass our function `process_partition` arguments and a
    # list to be processed. Be sure to check out https://pythonhosted.org/joblib/parallel.html#using-the-threading-backend
    # and experiment to determine if your function gains more speedup from
    # using mutiprocessing or threads

    results = Parallel(n_jobs=num_cores)(
        delayed(process_partition)(partition, connection_params) for partition in partitions)
    results = list(chain.from_iterable(results))
    pprint(len(results))


def main():
    """
    Test functionality
    """
    # client = connect()
    # user_mentions_map_reduce(client, 'twitter', 'subset_gu')

    # test_get_language_distribution(client)
    # test_get_language_subset(client)
    # create_lang_subset(client, 'twitter', 'gu')

    # test_get_top_k_users(client, 'twitter', ['ru'], USER_MENTIONS)
    # test_get_top_k_hashtags(client, 'twitter', ['ru'], HASHTAGS, 20)
    # get_hashtag_collection(client, 'twitter', 'hashtag_dist_en')

    # tweet_text = "hello world how are you"
    # pprint(([(create_ngrams(tweet_text.split(" "), i)) for i in range(1, 6)]))

    # sample_size = 100000
    # connection_params = ["twitter", "tweets"]
    # test_linear_scan(connection_params, sample_size)
    # parallel_test(cpu_count(), connection_params, sample_size)
if __name__ == '__main__':
    main()
