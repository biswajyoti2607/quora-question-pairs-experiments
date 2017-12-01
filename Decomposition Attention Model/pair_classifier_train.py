import numpy as np
import tensorflow as tf
import nltk
import time
import os.path
import pandas as pd
from tqdm import tqdm
from pair_classifier_model import Model
import io
import random as random


tf.logging.set_verbosity(tf.logging.INFO)

def getDataframe(path):
    temp_ls = []
    with io.open(path,mode='r',encoding='utf8') as fin:
        for line in enumerate(fin):
            tup_val = line[1]
            tup_array = tup_val.split('|*|*|*|')
            tup_fdat = [tup_array[3].strip(), tup_array[4].strip(), float(tup_array[5].strip())]
            temp_ls.append(tup_fdat)
        test_df = pd.DataFrame(temp_ls)
        test_df.columns = ['question1', 'question2', 'is_duplicate']

    return test_df


def add_embedded_sentences(df, word_vec_dict):
    """Here sentences are sanitized, tokenized, and vector representations of question are added to the dataframe."""
    q1_vec = []
    q2_vec = []
    for q1q2 in tqdm(zip(df['question1'], df['question2'])):
        if type(q1q2[0]) is not str:
            q1_clean = str(q1q2[0])
        else:
            q1_clean = nltk.word_tokenize(q1q2[0].encode('ascii', 'ignore').decode('utf-8', 'ignore').
                                          lower().replace('[^0-9a-zA-Z ]+', ''))

        sentence1_vecs = [word_vec_dict[w] for w in q1_clean if w in word_vec_dict]
        # This makes sure a sentence is at least represented by a zeros vector,
        # if none of the words where found in the dictionary.
        q1_vec.append(sentence1_vecs) if (len(sentence1_vecs) >= 1) \
            else q1_vec.append(np.zeros((1, 300), dtype=float))

        if type(q1q2[1]) is not str:
            q2_clean = str(q1q2[1])
        else:
            q2_clean = nltk.word_tokenize(q1q2[1].encode('ascii', 'ignore').decode('utf-8', 'ignore').
                                          lower().replace('[^0-9a-zA-Z ]+', ''))
        sentence2_vecs = [word_vec_dict[w] for w in q2_clean if w in word_vec_dict]
        q2_vec.append(sentence2_vecs) if (len(sentence2_vecs) >= 1) \
            else q2_vec.append(np.zeros((1, 300), dtype=float))

    df['question1_vecs'] = pd.Series(q1_vec)
    df['question2_vecs'] = pd.Series(q2_vec)
    return df


# Building/Loading training data
train_df_pickle_initial = r'snli_preprocessed.pkl'
test_df_pickle_initial = r'quora_valid.pkl'
#train_csv_initial = r'snli_sentence_data.tsv'
#test_csv_initial = r'Validation_stripped.tsv'

train_df_pickle_final = r'quora_train_enlarged.pkl'
test_df_pickle_final = r'quora_valid.pkl'
#train_csv_final = r'Training_stripped.tsv'
#test_csv_final = r'Validation_stripped.tsv'

word_embeddings = r'..\glove.840B.300d.txt'

if os.path.isfile(train_df_pickle_initial) and os.path.isfile(test_df_pickle_initial):
    # If pickled datasets are present, we load them and avoid pre-processing the data over again.
    print('loading pickles')
    test_df = pd.read_pickle(test_df_pickle_initial)
    train_df = pd.read_pickle(train_df_pickle_initial)
    test_df_final = pd.read_pickle(test_df_pickle_final)
    train_df_final = pd.read_pickle(train_df_pickle_final)
else:
    # If pickled datasets are not available, load and pre-process the CSVs
    print('processing & pickling CSVs')
    train_df = pd.read_csv(train_csv_initial, encoding='utf-8', sep='\t')
    test_df = getDataframe(test_csv_initial)
    train_df_final = getDataframe(train_csv_final)
    test_df_final = getDataframe(test_csv_final)
    train_df = train_df[['question1', 'question2', 'is_duplicate']]
    test_df = test_df[['question1', 'question2', 'is_duplicate']]
    train_df_final = train_df_final[['question1', 'question2', 'is_duplicate']]
    test_df_final = test_df_final[['question1', 'question2', 'is_duplicate']]

    print('building word-vec dictionary')
    with open(word_embeddings, encoding="utf8") as f:
        vec_dictionary = {}
        content = f.readline()
        for i in tqdm(range(100000)):
            content = f.readline()
            content = content.strip()
            content = content.split(' ')
            word = content.pop(0)
            vec_dictionary.update({word: [float(i) for i in content]})

    print('test_df add_embedded_sentences')
    test_df = add_embedded_sentences(test_df, vec_dictionary)
    test_df_final = add_embedded_sentences(test_df_final, vec_dictionary)
    print('train_df add_embedded_sentences')
    train_df = add_embedded_sentences(train_df, vec_dictionary)
    train_df_final = add_embedded_sentences(train_df_final, vec_dictionary)

    # We save the pickled dataframes to avoid pre-processing step everytime.
    print('pickling')
    test_df.to_pickle(test_df_pickle_initial)
    train_df.to_pickle(train_df_pickle_initial)
    test_df_final.to_pickle(test_df_pickle_final)
    train_df_final.to_pickle(train_df_pickle_final)
    print('pickling DONE')


log_dir = 'log'
save_dir = 'save'
model = Model('TRAIN', 'model')
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    # summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(log_dir, time.strftime('%Y-%m-%d-%H-%M-%S')))
    writer_eval = tf.summary.FileWriter(os.path.join(log_dir, 'eval', time.strftime('%Y-%m-%d-%H-%M-%S')))
    writer.add_graph(sess.graph)
    sess.run(init_op)
    saver = tf.train.Saver()

    ####################################################################################################################
    # code to pretrain model #
    ####################################################################################################################
    print("pretraining")
    for epoch in range(1):
        for i in tqdm(range(len(train_df))):
            # This loop runs the training data through the model one sentence pair at a time.
            a_feed = train_df['question1_vecs'][i]
            b_feed = train_df['question2_vecs'][i]
            # making sure a_feed and b_feed are at least one word long, otherwise we skip this sample
            if len(a_feed) < 1 or len(b_feed) < 1:
                continue
            label_feed = np.array([train_df['is_duplicate'][i]])

            summary, train_op, loss = sess.run([model.loss_summary, model.train_op, model.loss],
                                               {model.a: a_feed, model.b: b_feed, model.label: label_feed})
            if i % 1000 == 0:
                writer.add_summary(summary, global_step=(i + 1) * (1 + epoch))

            if i % 10000 == 0:
                # We are running validation data through the model every 50000 iterations of training
                print('\nRunning validation')
                # Resetting accuracy and iteration counter variables before running the validation set
                sess.run(tf.assign(model.accuracy, tf.constant(0.0)))
                sess.run(tf.assign(model.validation_iter, tf.constant(1)))
                random_sample = test_df.sample(n=2000, replace=True)
                correct = 0
                for j in (range(len(random_sample))):
                    a_feed = random_sample['question1_vecs'].values[j]
                    b_feed = random_sample['question2_vecs'].values[j]
                    label_feed = np.array([random_sample['is_duplicate'].values[j]])
                    is_duplicate_prediction, accuracy_summ, accuracy, my_iter = \
                        sess.run(
                            [model.classes, model.accuracy_summary_op, model.accuracy_op, model.validation_iter_op],
                            {model.a: a_feed, model.b: b_feed, model.label: label_feed})
                    if (is_duplicate_prediction[0] == label_feed[0]):
                        correct += 1.0
                print("Validation Accuracy" + str(correct / float(len(random_sample))))
                writer_eval.add_summary(accuracy_summ, global_step=(i + 1) * (1 + epoch))

    ####################################################################################################################
    # do the training for our model - this comes AFTER the pretraining #
    ####################################################################################################################
    print("actual training")
    wrongTest = [] #pd.DataFrame(columns=['question1', 'question2', 'is_duplicate'])
    currentList = []

    for epoch in range(10): #50
        print ("Currently in epoch :",epoch)

        if (epoch >= 1):
            for i,z in zip(tqdm(currentList),range(len(currentList))):
                # This loop runs the training data through the model one sentence pair at a time.
                a_feed = train_df_final['question1_vecs'][i]
                b_feed = train_df_final['question2_vecs'][i]
                # making sure a_feed and b_feed are at least one word long, otherwise we skip this sample
                if len(a_feed) < 1 or len(b_feed) < 1:
                    continue
                label_feed = np.array([train_df_final['is_duplicate'][i]])

                summary, train_op, loss = sess.run([model.loss_summary, model.train_op, model.loss],
                                                   {model.a: a_feed, model.b: b_feed, model.label: label_feed})
                if z % 1000 == 0:
                    writer.add_summary(summary, global_step=(z + 1) * (1 + epoch))

                is_duplicate_prediction, accuracy_summ, accuracy, my_iter = \
                    sess.run([model.classes, model.accuracy_summary_op, model.accuracy_op, model.validation_iter_op],
                             {model.a: a_feed, model.b: b_feed, model.label: label_feed})

                if is_duplicate_prediction[0] != label_feed[0]:
                    wrongTest.append(i)
                    # df2 = pd.DataFrame({'question1': [a_feed], 'question2': [b_feed], 'is_duplicate': [label_feed[0]]})
                    # wrongTest.append(df2)

                if z % 10000 == 0:
                    # We are running validation data through the model every 50000 iterations of training
                    print('\nRunning validation')
                    # Resetting accuracy and iteration counter variables before running the validation set
                    sess.run(tf.assign(model.accuracy, tf.constant(0.0)))
                    sess.run(tf.assign(model.validation_iter, tf.constant(1)))
                    random_sample = test_df_final.sample(n=2000, replace=True)
                    correct_classification = 0
                    for j in (range(len(random_sample))):
                        # this is where the question is. find all the ones that are wrong
                        a_feed = random_sample['question1_vecs'].values[j]
                        b_feed = random_sample['question2_vecs'].values[j]
                        label_feed = np.array([random_sample['is_duplicate'].values[j]])
                        is_duplicate_prediction, accuracy_summ, accuracy, my_iter = \
                            sess.run(
                                [model.classes, model.accuracy_summary_op, model.accuracy_op, model.validation_iter_op],
                                {model.a: a_feed, model.b: b_feed, model.label: label_feed})
                        if (is_duplicate_prediction[0] == label_feed[0]):
                            correct_classification = correct_classification + 1

                    accuracy = correct_classification / len(random_sample)
                    print ("Validation accuracy :", accuracy)
                    writer_eval.add_summary(accuracy_summ, global_step=(z + 1) * (1 + epoch))

                    if(accuracy > 0.85):
                        checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch)
                        print('model saved to {}'.format(checkpoint_path))

        else:
            for i in tqdm(range(len(train_df_final))):#tqdm(range(10000)):
                # This loop runs the training data through the model one sentence pair at a time.
                a_feed = train_df_final['question1_vecs'][i]
                b_feed = train_df_final['question2_vecs'][i]
                # making sure a_feed and b_feed are at least one word long, otherwise we skip this sample
                if len(a_feed) < 1 or len(b_feed) < 1:
                    continue
                label_feed = np.array([train_df_final['is_duplicate'][i]])

                summary, train_op, loss = sess.run([model.loss_summary, model.train_op, model.loss],
                                                   {model.a: a_feed, model.b: b_feed, model.label: label_feed})
                if i % 1000 == 0:
                    writer.add_summary(summary, global_step=(i + 1) * (1 + epoch))

                is_duplicate_prediction, accuracy_summ, accuracy, my_iter = \
                            sess.run([model.classes, model.accuracy_summary_op, model.accuracy_op, model.validation_iter_op],
                            {model.a: a_feed, model.b: b_feed, model.label: label_feed})

                if is_duplicate_prediction[0] != label_feed[0]:
                    wrongTest.append(i)
                    # df2 = pd.DataFrame({'question1': [a_feed], 'question2': [b_feed], 'is_duplicate': [label_feed[0]]})
                    # wrongTest.append(df2)

                if i % 10000 == 0:
                    # We are running validation data through the model every 50000 iterations of training
                    print('\nRunning validation')
                    # Resetting accuracy and iteration counter variables before running the validation set
                    sess.run(tf.assign(model.accuracy, tf.constant(0.0)))
                    sess.run(tf.assign(model.validation_iter, tf.constant(1)))
                    random_sample = test_df_final.sample(n=2000, replace=True)
                    correct_classification = 0
                    for j in (range(len(random_sample))):
                        #this is where the question is. find all the ones that are wrong
                        a_feed = random_sample['question1_vecs'].values[j]
                        b_feed = random_sample['question2_vecs'].values[j]
                        label_feed = np.array([random_sample['is_duplicate'].values[j]])
                        is_duplicate_prediction, accuracy_summ, accuracy, my_iter = \
                            sess.run([model.classes, model.accuracy_summary_op, model.accuracy_op, model.validation_iter_op],
                                     {model.a: a_feed, model.b: b_feed, model.label: label_feed})
                        if (is_duplicate_prediction[0]==label_feed[0]):
                            correct_classification=correct_classification+1

                    accuracy=correct_classification/len(random_sample)
                    print ("Validation accuracy :",accuracy)
                    writer_eval.add_summary(accuracy_summ, global_step=(i + 1) * (1 + epoch))

        # # Saving a checkpoint after each epoch
        # checkpoint_path = os.path.join(save_dir, 'model.ckpt')
        # saver.save(sess, checkpoint_path, global_step=epoch)
        # print('model saved to {}'.format(checkpoint_path))

        nextList = random.sample(wrongTest, int(len(wrongTest) * 0.65))
        addMore = len(train_df_final) - len(nextList)
        addList = random.sample(range(len(train_df_final)), addMore)
        currentList = nextList + addList
