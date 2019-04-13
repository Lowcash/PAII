// file:	dynArray.h
// version: 3.0
// summary:	Dynamic Array
// author:  Petr Gajdoš
// Copyright (c) 2009-2014 All Rights Reserved

#ifndef __DYNARRAY_
#define  __DYNARRAY_

#include "stdafx.h"

template<typename T>
class DynArray
{

private:
	const static unsigned int initCapacity;		// initial capacity of array memory (elements)
	const static float resizeFactor;			// multiplier (enlarge array)

	T*				data;
	unsigned int	capacity;
	unsigned int	count;

public:
	DynArray(unsigned int _capacity = initCapacity);
	DynArray(unsigned int unCount, const T& val);
	DynArray(unsigned int unCount, const T* _data);
	DynArray(const DynArray<T>& a);
	~DynArray();

	DynArray& operator=(const DynArray& a);

	inline const T& operator[](unsigned int unIndex) const { assert(unIndex < count &&"DynArray::[] - Index out of bounds"); return data[unIndex]; }
	inline const T& at(unsigned int unIndex) const { assert(unIndex < count &&"DynArray::[] - Index out of bounds"); return data[unIndex]; }

	inline T& operator[](unsigned int unIndex) { assert(unIndex < count &&"DynArray::[] - Index out of bounds"); return data[unIndex]; }
	inline T& at(unsigned int unIndex) { assert(unIndex < count &&"DynArray::[] - Index out of bounds"); return data[unIndex]; }

	void push_back(const T& item);
	//T pop_back();
	void pop_back();
	void insert_at_end(const T* p, unsigned int uCount);
        void insert_at(unsigned int position, const T& item);
	void swap(unsigned int from, unsigned int to);

	void enlarge();
	void reserve(unsigned int unNewCapacity);
	void resize(unsigned int unNewSize);
	void resize(unsigned int unNewSize, const T& newElementsFromThis);

	void clear();
	void erase(unsigned int unIndex);
	void minimize();

	inline unsigned int size() const { return count; }
	inline T* front() const { return data; }
	inline T* back() const { assert(0 < count &&"DynArray::[] - No data"); return &data[count-1]; }
	inline bool	empty() const { return count == 0; }
	inline T* dataPtr(unsigned int unIndex) const { assert(unIndex < count &&"DynArray::[] - Index out of bounds"); return &data[unIndex]; }
};

template<typename T> const unsigned int DynArray<T>::initCapacity = (1<<20);  //cca 1M values
template<typename T> const float DynArray<T>::resizeFactor = 2.0f;

template<typename T> DynArray<T>::DynArray(unsigned int _capacity) 
	:capacity(_capacity), count(0)
{
	data = (T*)malloc(capacity*sizeof(T));
	//data = (T*)calloc(capacity,sizeof(T));
}

template<typename T> DynArray<T>::DynArray(unsigned int unCount, const T& val)
	:capacity(unCount),count(unCount)
{
	data = (T*)malloc(capacity*sizeof(T));
	while(unCount--)
		memcpy(data + unCount,&val,sizeof(T));
}

template<typename T> DynArray<T>::DynArray(unsigned int unCount, const T* _data)
	:capacity(unCount),count(unCount)
{
	data = (T*)malloc(capacity*sizeof(T));
	memcpy(data,_data,capacity*sizeof(T));
}

template<typename T> DynArray<T>::DynArray(const DynArray<T>& a)
	:capacity(a.capacity),count(a.count)
{
	if(this != &a)
	{
		data = (T *)malloc(sizeof(T)*a.capacity);
		memcpy(data, a.data, sizeof(T)*a.count);
	}
}

template<typename T>DynArray<T>::~DynArray() 
{ 
	if(data) 
	{ 
		free(data); 
		data = 0; 
	} 
}

template<typename T>DynArray<T>& DynArray<T>::operator=(const DynArray &a)
{
	if(this != &a)
	{
		if(a.count == 0) clear();
		else
		{
			if (capacity != a.capacity)
				resize(a.capacity);
			//resize(a.count);
			count = a.count;
			memcpy(data, a.data, sizeof(T)*a.count);
		}
	}
	return *this;
}

template<typename T>void DynArray<T>::enlarge()
{
	capacity = (unsigned int)(capacity * resizeFactor);
	data = (T *)realloc(data, sizeof(T)*capacity);
}

template<typename T>void DynArray<T>::resize(unsigned int unNewSize)
{
	if(unNewSize)
	{
		if((unNewSize > capacity) || (unNewSize < capacity/2))
		{
			capacity = unNewSize;
			data = (T *)realloc(data, sizeof(T)*capacity);	
		}
	}
	else
		clear();
	count = unNewSize;
}

template<typename T>void DynArray<T>::resize(unsigned int unNewSize, const T& newElementsFromThis)
{
	if(unNewSize)
	{
		if((unNewSize > capacity) || (unNewSize < capacity/2))
		{
			capacity = unNewSize;
			data = (T *)realloc(data, sizeof(T)*capacity);	
		}
		for(unsigned int i = count; i < unNewSize; ++i) data[i] = newElementsFromThis;
	}
	else
		clear();
	count = unNewSize;
}

template<typename T>void DynArray<T>::minimize()
{
	capacity = count;
	data = (T *)realloc(data, sizeof(T)*capacity);
}

template<typename T>void DynArray<T>::erase(unsigned int unIndex)
{
	if(count == 1)
		clear();
	else
	{
		assert(unIndex < count && "DynArray<>::Erase(unsigned int) - Index out of bounds");
		memmove(data + unIndex, data + unIndex + 1, (count - 1 - unIndex) * sizeof(T) );
		count--;
	}
}

template<typename T>void DynArray<T>::pop_back()
{
	if (count == 0) return;
	if(count == 1)
	{
		clear();
	}
	else
	{
		--count;
	}
}

template<typename T>void DynArray<T>::clear()
{
	if(0 == count) return;
	data = (T *)realloc(data, sizeof(T)*initCapacity); 
	capacity = initCapacity;
	count = 0;
}

template<typename T>void DynArray<T>::push_back(const T& item)
{
	count++;
	if(count > capacity)
	{
		capacity = (unsigned int)(capacity * resizeFactor);
		data = (T *)realloc(data, sizeof(T)*capacity);
	}
	memcpy(data + count - 1, &item, sizeof(T) );
	//data[count - 1] = item;
}

template<typename T>void DynArray<T>::reserve(unsigned int unNewCapacity)
{
	capacity = unNewCapacity;
	data = (T *)realloc(data, sizeof(T)*capacity);
	if(count > capacity) count = capacity;
}

template<typename T>void DynArray<T>::insert_at_end(const T* p, unsigned int uCount)
{
	unsigned int uNewSize = count + uCount;
	if(uNewSize > capacity)
	{
		capacity = (unsigned int)(capacity * resizeFactor);
		if(capacity < uNewSize) capacity = uNewSize;
		data = (T *)realloc(data, sizeof(T)*capacity);
	}
	//for(unsigned int i = 0; i < uCount; ++i) data[count + i] = p[i];
	memcpy(&data[count], &p[0], uCount*sizeof(T));
	count += uCount;
}

template<typename T>void DynArray<T>::swap(unsigned int from, unsigned int to)
{
	T tmp = data[to];
	data[to] = data[from];
	data[from] = tmp;
}

template<typename T>void DynArray<T>::insert_at(unsigned int position, const T& item)
{
	if (position > count) return;
	if (position == count)
	{
		push_back(item);
		return;
	}

	unsigned int uNewSize = count + 1;
	if (uNewSize > capacity)
	{
		capacity = (unsigned int)(capacity * resizeFactor);
		if (capacity < uNewSize) capacity = uNewSize;
		data = (T *)realloc(data, sizeof(T)*capacity);
	}

	for (unsigned int i = count; i > position; i--)		//shift right
	{
		swap(i, i - 1);
	}
	
	data[position] = item;

	count += 1;
}


#endif
