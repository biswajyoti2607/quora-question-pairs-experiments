import tensorflow as tf
import duplicate_questions.data.data_manager import DataManager
with tf.Session() as sess:
	logger.info("Getting latest checkpoint in {}".format(model_load_dir))
	last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
	logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
	saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
	saver.restore(sess, last_checkpoint)
	logger.info("Successfully loaded {}!".format(last_checkpoint))

	# Get a generator of test batches
	test_batch_gen = DataManager.get_batch_generator(
		get_test_instance_generator, batch_size)

	y_pred = []
	for batch in tqdm(test_batch_gen,
					  total=num_test_steps,
					  desc="Test Batches Completed"):
		feed_dict = self._get_test_feed_dict(batch)
		y_pred_batch = sess.run(self.y_pred, feed_dict=feed_dict)
		y_pred.append(y_pred_batch)
	y_pred_flat = np.concatenate(y_pred, axis=0)