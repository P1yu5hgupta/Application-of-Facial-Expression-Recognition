import React from "react"
import styles from "./Cards.module.css"
import Card from "./Card/Card"
const Cards = (props)=>{
  const cards = props.data.emotion.map((element, index) => (
    <Card data = {element} key = {index}></Card>
  ));
  return(
    <div className = {styles.Cards}>
      {cards}
    </div>
  )
}
export default Cards