import os
import sys
import asyncio
import json
from datetime import datetime

import pandas as pd
from pyensign.events import Event
from pyensign.ensign import Ensign
from river.compose import Pipeline
from river import preprocessing, linear_model, metrics





async def handle_ack(ack):
    ts = datetime.fromtimestamp(ack.committed.seconds + ack.committed.nanos / 1e9)
    print(ts)

async def handle_nack(nack):
    print(f"Could not commit event {nack.id} with error {nack.code}: {nack.error}")

 
class AccidentDataPublisher:
    def __init__(self, topic="river_pipeline", interval=1):
        self.topic = topic
        self.interval = interval
        self.ensign = Ensign()

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.publish())

    async def publish(self):
        """
        Read data from the dataset and publish to river_pipeline topic.
        """
        # create the topic if it does not exist
        await self.ensign.ensure_topic_exists(self.topic)
        train_df = pd.read_csv(os.path.join("data", "{ADD DATA SOURCE}.csv"))  ###### <- ADD DATA SOURCE 
        train_dict = train_df.to_dict("records")
        for record in train_dict:
            print(record)
            event = Event(json.dumps(record).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)

        # In this example, we have reached the end of the file, so we will send a "done" message
        # and the subscriber will use this message as a notification to print out the final statistics.
        # In a production environment, this stream would be open forever, sending messages to the
        # subscriber and the model and metrics will continually get updated.
        event = Event(json.dumps({"done":"yes"}).encode("utf-8"), mimetype="application/json")
        await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)
        await asyncio.sleep(self.interval)   


class AccidentDataSubscriber:
    """
    The AccidentDataSubscriber class reads from the river_pipeline topic and incrementally learns
    from the data until it has learned from all of the instances.  It publishes the precision
    and recall metrics to the river_metrics topic after they are calculated at each step.
    """

    def __init__(self, sub_topic="river_pipeline", pub_topic="river_metrics", interval=1):
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.interval = interval
        self.ensign = Ensign()
        self.initialize_model_and_metrics()

    def run(self):
        """
        Run the subscriber forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    def initialize_model_and_metrics(self):
        """
        Initialize a river model and set up metrics to evaluate the model as it learns
        """
        self.model = Pipeline(
            integer = compose.Select('injuries_total','injuries_fatal', 'injuries_incapacitating', 'injuries_non_incapacitating', 'num_units', 'age') | preprocessing.StandardScaler()
            categorical = compose.Select('damage', 'street_direction', 'crash_hour', 'crash_day_of_week', 'crash_month') | preprocessing.OneHotEncoder(drop_first=True)
            model = (integer + categorical) | linear_model.LogisticRegression() # logistic model to run 
        )
        self.confusion_matrix = metrics.ConfusionMatrix(classes=[0,1])
        self.classification_report = metrics.ClassificationReport()
        self.precision_recall =  metrics.Precision(cm=self.confusion_matrix, pos_val=0) + metrics.Recall(cm=self.confusion_matrix, pos_val=0)

    async def run_model_pipeline(self, event):
        """
        Make a prediction and update metrics based on the predicted value and the actual value
        Incrementally learn/update model based on the actual value
        Continue until "done" message is received
        """
        record = json.loads(event.data)
        print(record)
        if "done" not in record.keys():
            y_pred = self.model.predict_one(record["TARGET"])
            if y_pred is not None:
                self.confusion_matrix.update(y_true=record["TARGET"], y_pred=y_pred)
                self.classification_report.update(y_true=record["TARGET"], y_pred=y_pred)
            # the precision and recall won't be great at first, but as the model learns on
            # new data, the scores improve
            print(self.precision_recall)
            pr_list = self.precision_recall.get()
            pr_dict = {"precision": pr_list[0], "recall": pr_list[1]}
            event = Event(json.dumps(pr_dict).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.pub_topic, event, on_ack=handle_ack, on_nack=handle_nack)
            # learn from the train example and update the model
            self.model = self.model.learn_one(record["TARGET"]) 
        else:
            # We are printing out the final metrics here because we have looped through all of 
            # the records.
            print("Final Metrics", self.precision_recall)
            print(self.classification_report)
            print(self.confusion_matrix)

    async def subscribe(self):
        """
        Receive messages from river_pipeline topic
        """

        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.sub_topic)
        await self.ensign.ensure_topic_exists(self.pub_topic)

        async for event in self.ensign.subscribe(self.sub_topic):
            await self.run_model_pipeline(event)
        

class MetricsSubscriber:
    """
    The MetricsSubscriber class reads from the river_metrics topic and checks to see
    if the precision and recall have fallen below a specified threshold and prints to screen.
    This code can be extended to update a dashboard and/or send alerts.
    """

    def __init__(self, topic="river_metrics", threshold=0.60, interval=1):
        self.topic = topic
        self.interval = interval
        self.threshold = threshold
        self.ensign = Ensign()

    def run(self):
        """
        Run the subscriber forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    async def check_metrics(self, event):
        """
        Check precision and recall metrics and print if below threshold
        """
        metric_info = json.loads(event.data)
        precision = metric_info["precision"]
        recall = metric_info["recall"]
        if precision < self.threshold:
            print(f"Precision is below threshold: {precision}")
        if recall < self.threshold:
            print(f"Recall is below threshold: {recall}")

    async def subscribe(self):
        """
        Receive messages from river_train_data topic
        """

        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.topic)

        async for event in self.ensign.subscribe(self.topic):
            await self.check_metrics(event)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "publish":
            publisher = AccidentDataPublisher()
            publisher.run()
        elif sys.argv[1] == "subscribe":
            subscriber = AccidentDataSubscriber()
            subscriber.run()
        elif sys.argv[1] == "metrics":
            subscriber = MetricsSubscriber()
            subscriber.run()
        else:
            print("Usage: python accident_model.py [publish|subscribe|metrics]")
    else:
        print("Usage: python accident_model.py [publish|subscribe|metrics]")