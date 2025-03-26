import React, {useState} from 'react';
import { useRouter } from 'next/router';
import jsonData from '../../data/articles/Dexter_Reed-2024_04_10-foody.json'
import { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async (context) => {
  const { email } = context.query;

  if (!email) return { props: { user: null } };

  const res = await fetch(`http://localhost:3000/api/loadArticles`); // load from loadArticles.ts API route
  let articleProp = await res.json();
  const articles = articleProp['message'];

  return { props: { articles }};
};

export default function Dashboard({articles}) {
    // Define state with TypeScript types
    const router = useRouter();
    const { email } = router.query;
    const articleArray = JSON.parse(articles);


    return (
        <div>
            {articleArray.map((article, index) => (
                <a href="https://www.w3schools.com">{article}</a> 
            ))}
        </div>
    )
};