import React from "react"
import styles from "./Card.module.css"


class CardBody extends React.Component {
  render() {
    console.log(styles.card)
    return (
      <div className={styles.cardBody}>
        <p className={styles.date}>Track Start Time::{this.props.data.startTime}</p>
        <p className={styles.date}>Track End Time::{this.props.data.endTime}</p>
        <h2>Emotions</h2>
        <p className={styles.bodyContentHappy}>Happy: {this.props.data.happy}%</p>
        <p className={styles.bodyContentSad}>Sad: {this.props.data.sad}%</p>
        <p className={styles.bodyContentAngry}>Angry: {this.props.data.angry}%</p>
        <p className={styles.bodyContentDisgust}>Disgust: {this.props.data.disgust}%</p>
        <p className={styles.bodyContentSurprised}>Surprised: {this.props.data.surprised}%</p>
        <p className={styles.bodyContentFearfull}>Fearfull: {this.props.data.fearfull}%</p>
        <p className={styles.bodyContentNeutral}>Neutral: {this.props.data.neutral}%</p>
      </div>
    )
  }
}

class Card extends React.Component {
  render() {
    return (
      <article className={styles.card}>
        <CardBody data = {this.props.data}/>
      </article>
    )
  }
}

export default Card