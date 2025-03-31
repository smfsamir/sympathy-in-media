import React, {useState} from 'react';
import { useRouter } from 'next/router';
import { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async (context) => {
  const { email } = context.query;
  if (!email) return { props: { user: null } };

  const res = await fetch(`http://localhost:3000/api/loadArticles`); // TODO: change on deployment
  let articleProp = await res.json();
  const articles = articleProp['message'];

  return { props: { articles }};
};



export default function Dashboard({articles}) {
    // Define state with TypeScript types
    const router = useRouter();
    const { email } = router.query; // this grabs the email from the URL
    const articleArray = JSON.parse(articles);
    
    const openArticle = (article) => {
        router.push(`/article_annotation/?email=${email}&article=${article}`); // open apporpriate article
    };

    return (
        <div>
            {articleArray.map((article, index) => (
                <button onClick={() => openArticle(article)}>{article}</button>
            ))}
        </div>
    )
};